"""
Backend processing logic for Manga Extract-and-Clean.
No GUI imports — all functions accept a progress_callback for status updates.

Uses vendored manga_translator modules directly (detection, OCR, inpainting)
instead of the MangaTranslator orchestrator class.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image

from manga_translator.config import Config, Detector, Ocr, Inpainter
from manga_translator.utils.generic import Context, load_image
from manga_translator.detection import dispatch as dispatch_detection, prepare as prepare_detection
from manga_translator.ocr import dispatch as dispatch_ocr, prepare as prepare_ocr
from manga_translator.textline_merge import dispatch as dispatch_textline_merge
from manga_translator.mask_refinement import dispatch as dispatch_mask_refinement
from manga_translator.inpainting import dispatch as dispatch_inpainting, prepare as prepare_inpainting

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


# ---------------------------------------------------------------------------
# JSON operations (adapted from C:\git\process-json)
# ---------------------------------------------------------------------------

def combine_jsons(folder: Path, delete_originals: bool = True) -> Path:
    """Combine per-image .json files into lmt_raw.json, then delete originals."""
    output_path = folder / "lmt_raw.json"
    combined = []
    processed_files = []

    for f in sorted(folder.glob("*.json")):
        if f.name.startswith("lmt_"):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            combined.append({"file": f.name, "content": data})
            processed_files.append(f)
        except Exception as e:
            print(f"Warning: failed to read {f.name}: {e}")

    output_path.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if delete_originals:
        for f in processed_files:
            f.unlink()

    return output_path


def compress_json(raw_path: Path) -> Path:
    """Extract only the text (5th element) from each row -> lmt_script.txt."""
    output_path = raw_path.parent / "lmt_script.txt"

    data = json.loads(raw_path.read_text(encoding="utf-8"))
    lines = []
    for entry in data:
        content = entry.get("content", [])
        if isinstance(content, list):
            for row in content:
                if isinstance(row, list) and len(row) >= 5:
                    # Escape newlines as \n
                    text = str(row[4]).replace('\n', '\\n')
                    lines.append(text)

    output_path.write_text('\n'.join(lines), encoding="utf-8")
    return output_path


def script_to_json(original_path: str, translated_script_path: str, output_dir: str):
    """Convert translated script (txt, one line per text) back into original JSON structure.

    Returns (output_path, replacement_count, warnings).
    """
    orig_p = Path(original_path)
    trans_p = Path(translated_script_path)
    out_dir = Path(output_dir)

    output_filename = f"lmt_{trans_p.stem}_converted.json"
    output_path = out_dir / output_filename

    original_data = json.loads(orig_p.read_text(encoding="utf-8"))

    # Read translated script as lines
    translated_lines = trans_p.read_text(encoding="utf-8").strip().split('\n')
    # Unescape \n back to actual newlines
    translated_list = [line.replace('\\n', '\n') for line in translated_lines]

    if not isinstance(original_data, list):
        raise ValueError("Original JSON must contain a top-level array.")

    translation_iter = iter(translated_list)
    replacement_count = 0

    for entry in original_data:
        content = entry.get("content", [])
        if isinstance(content, list):
            for row in content:
                if isinstance(row, list) and len(row) >= 5:
                    try:
                        row[4] = next(translation_iter)
                        replacement_count += 1
                    except StopIteration:
                        break

    output_path.write_text(
        json.dumps(original_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    remaining = sum(1 for _ in translation_iter)
    warnings = []
    if remaining > 0:
        warnings.append(f"Translated file had {remaining} extra entries that were not used.")

    return str(output_path), replacement_count, warnings


def uncompress_json(original_path: str, translated_path: str, output_dir: str):
    """Restore translated text back into the original JSON structure (legacy).

    Returns (output_path, replacement_count, warnings).
    """
    # Delegate to script_to_json for compatibility
    return script_to_json(original_path, translated_path, output_dir)


# ---------------------------------------------------------------------------
# Manga image processing pipeline
# ---------------------------------------------------------------------------

def _list_images(folder: Path) -> list[Path]:
    """Return sorted list of image files in folder."""
    return [f for f in sorted(folder.iterdir())
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]


def _get_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


async def _process_single_image(config: Config, device: str, image_path: Path, output_folder: Path) -> str:
    """Run detection + OCR + textline merge + inpainting on one image.

    Returns a status string: 'processed', 'skipped', or 'skipped-partial'.
    """
    json_path = output_folder / f"{image_path.stem}.json"
    out_image_path = output_folder / f"{image_path.stem}{image_path.suffix}"
    lmt_raw_path = output_folder / "lmt_raw.json"

    has_json = json_path.exists() or lmt_raw_path.exists()
    has_image = out_image_path.exists()

    # Both outputs exist — skip entirely
    if has_json and has_image:
        return 'skipped'

    img = Image.open(image_path)
    img.verify()
    img = Image.open(image_path)
    img_rgb, img_alpha = load_image(img)

    # Need OCR (json doesn't exist yet and lmt_raw.json doesn't exist yet)
    text_regions = None
    mask_raw = None
    if not has_json:
        # --- Detection ---
        textlines, mask_raw, mask = await dispatch_detection(
            config.detector.detector, img_rgb,
            config.detector.detection_size, config.detector.text_threshold,
            config.detector.box_threshold, config.detector.unclip_ratio,
            config.detector.det_invert, config.detector.det_gamma_correct,
            config.detector.det_rotate, config.detector.det_auto_rotate,
            device, False,
        )

        if not textlines:
            json_path.write_text('[]', encoding='utf-8')
            if not has_image:
                img.save(out_image_path)
            return 'processed'

        # --- OCR ---
        textlines = await dispatch_ocr(
            config.ocr.ocr, img_rgb, textlines, config.ocr, device, False,
        )
        textlines = [t for t in textlines if t.text.strip()]

        if not textlines:
            json_path.write_text('[]', encoding='utf-8')
            if not has_image:
                img.save(out_image_path)
            return 'processed'

        # --- Textline merge ---
        text_regions = await dispatch_textline_merge(
            textlines, img_rgb.shape[1], img_rgb.shape[0], verbose=False,
        )

        # --- Save text regions as JSON ---
        json_str = '[\n'
        for region in text_regions:
            x1 = int(region.lines[..., 0].min())
            y1 = int(region.lines[..., 1].min())
            x2 = int(region.lines[..., 0].max())
            y2 = int(region.lines[..., 1].max())
            text = region.text.replace('\\', '\\\\').replace('"', '\\"')
            json_str += f'[{x1}, {y1}, {x2}, {y2}, "{text}"],\n'
        if text_regions:
            json_str = json_str[:-2] + '\n]'
        else:
            json_str = '[]'
        json_path.write_text(json_str, encoding='utf-8')

    # Need inpainting (cleaned image doesn't exist yet)
    if not has_image:
        # If OCR was skipped (json already existed), we still need detection for the mask
        if text_regions is None or mask_raw is None:
            textlines, mask_raw, mask = await dispatch_detection(
                config.detector.detector, img_rgb,
                config.detector.detection_size, config.detector.text_threshold,
                config.detector.box_threshold, config.detector.unclip_ratio,
                config.detector.det_invert, config.detector.det_gamma_correct,
                config.detector.det_rotate, config.detector.det_auto_rotate,
                device, False,
            )
            if textlines:
                textlines = await dispatch_ocr(
                    config.ocr.ocr, img_rgb, textlines, config.ocr, device, False,
                )
                textlines = [t for t in textlines if t.text.strip()]
                if textlines:
                    text_regions = await dispatch_textline_merge(
                        textlines, img_rgb.shape[1], img_rgb.shape[0], verbose=False,
                    )

        if text_regions and mask_raw is not None:
            try:
                mask = await dispatch_mask_refinement(
                    text_regions, img_rgb, mask_raw, 'fit_text',
                    config.mask_dilation_offset, config.ocr.ignore_bubble, False, config.kernel_size,
                )
                img_inpainted = await dispatch_inpainting(
                    config.inpainter.inpainter, img_rgb, mask, config.inpainter,
                    config.inpainter.inpainting_size, device, False,
                )
                inpainted_bgr = cv2.cvtColor(img_inpainted, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_image_path), inpainted_bgr)
            except Exception as e:
                print(f"Warning: inpainting failed for {image_path.name}: {e}")
                img.save(out_image_path)
        else:
            img.save(out_image_path)

    return 'processed'


async def _async_process_folder(
    input_folder: str,
    output_folder: str,
    progress_callback: Callable,
):
    """Async core: process all images, then combine and compress JSONs."""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = _list_images(input_path)
    if not image_files:
        raise FileNotFoundError("No image files found in the selected folder.")

    total = len(image_files)
    device = _get_device()
    config = Config()

    # Pre-download models
    progress_callback("Downloading/loading models...", 0, total)
    await prepare_detection(config.detector.detector)
    await prepare_ocr(config.ocr.ocr, device)
    await prepare_inpainting(config.inpainter.inpainter, device)

    failures = []
    skipped = 0
    for i, image_path in enumerate(image_files):
        progress_callback(f"Processing {image_path.name}", i, total)
        try:
            status = await _process_single_image(config, device, image_path, output_path)
            if status == 'skipped':
                skipped += 1
        except Exception as e:
            failures.append((image_path.name, str(e)))
            print(f"Error processing {image_path.name}: {e}")

    if skipped > 0:
        print(f"Skipped {skipped} already-processed image(s)")

    # Combine + compress (skip if lmt_raw.json already exists)
    raw_path = output_path / "lmt_raw.json"
    if not raw_path.exists():
        progress_callback("Combining JSON files...", total, total)
        raw_path = combine_jsons(output_path)

        progress_callback("Compressing JSON...", total, total)
        compress_json(raw_path)

    progress_callback("Done", total, total)
    return failures


def process_folder(
    input_folder: str,
    output_folder: str,
    progress_callback: Callable,
) -> list:
    """Entry point for the worker thread. Returns list of (filename, error) failures."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            _async_process_folder(input_folder, output_folder, progress_callback)
        )
    finally:
        loop.close()
