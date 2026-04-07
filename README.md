# Lightweight Manga Typeset

A fully client-side web app for typesetting translated text onto manga/comic pages. Features built-in OCR text extraction and AI inpainting — all running in the browser via ONNX Runtime Web. No server, no install, no build step.

## Recommend workflow:

Load raw manga -> Extract raw textboxes + clean raw text -> Export raw script -> Put raw script to a LLM like Grok, Gemini to translate them to desired language -> Make sure the translated script have the same lines count as the raw script (if lines count mismatch, feed LLM less lines each chat) -> Import the translated script -> Edit and proofread the manga -> Save the translated manga.

You can co-op with others or edit on multiple devices by saving the cleaned manga without text (disable text boxes visibility). Use export and import .json to share the text boxes coord and content.

## Getting Started

1. Go to https://zuttodos.github.io/lightweight-manga-typeset/ or open `index.html` in a modern browser (Chrome 113+ recommended for WebGPU acceleration).
2. Click **Load** to select image files (optionally paired with JSON bounding-box data).
3. Click **Extract** to automatically detect text, OCR it, and clean the images using AI inpainting.
4. Use **Add** to create new text boxes, or edit existing ones.
5. Use **Brush** to paint over areas, or **Inpaint** brush to AI-remove regions.
6. Click **Save** to render everything onto images and download them.

> **Note:** Extract and Inpaint features require serving via HTTP (e.g. `python -m http.server 8000`) due to ONNX model loading. The rest of the app works with `file://`.

## Loading Files

| Input | What happens |
|---|---|
| Images only (`.jpg`, `.png`, `.webp`, etc.) | Loaded with no text boxes. Use **Add** to create them. |
| Images + matching JSON (`page1.jpg` + `page1.json`) | Text boxes loaded from the JSON bounding-box data. |
| Bundle JSON (`[{file, content}, ...]`) + images | Each entry maps a filename (with or without extension) to its bounding boxes. |

JSON format per file: an array of `[x0, y0, x1, y1, "text"]` entries. Coordinates can be normalized (0-1) or pixel values.

## Toolbar Reference

### File Operations

| Button | Action |
|---|---|
| **Load** | Open file picker to load images and optional JSON files. |
| **Save** | Render images with strokes at full resolution. When **Text Boxes** is enabled, text boxes are also rendered. Only saves images visible in the current view mode. |
| **Export .json** | Export all bounding boxes to a bundle JSON file (`lmt_textbox.json`). File names are base names without extension. Coordinates are normalized (0-1). |
| **Import .json** | Import bounding boxes from one or more bundle JSON files, replacing current bounding boxes for matching images. Matches by base name (extension-agnostic). |
| **Export script** | Export text from all visible bounding boxes to `lmt_script.txt` — one line per box. Newlines within text are escaped as `\n`. |
| **Import script** | Import text lines from a `.txt` file. Line count must match total bounding box count. Each line replaces the corresponding box's text. |

### AI Features

| Button | Action |
|---|---|
| **Extract** | Run AI text detection + OCR on visible images. Merges textlines into speech bubbles, creates bounding boxes with detected text, and inpaints text regions on the stroke canvas (original image is preserved). Shows a confirmation dialog before running. |
| **Inpaint pad** (number input) | Controls mask expansion for Extract inpainting (pixels per 1000px of image size). Default: 10. Not applied to Inpaint brush. |

### Adding Text Boxes

| Button | Shortcut | Action |
|---|---|---|
| **Add** | **N** | Toggle Add mode. While active, left-click and drag on any image to draw a new bounding box. A text editor appears on release. |
| **Add+** | | Toggle multi-add. When on, Add mode stays active after creating a box. When off (default), Add mode exits after one box is created. |

### Brush & Eraser

| Control | Shortcut | Action |
|---|---|---|
| **Brush** | **B** | Toggle Brush tool. Left-click and drag on the image to paint. Strokes render on a canvas layer on top of the image. |
| **Eraser** | **E** | Toggle Eraser tool. Left-click and drag to erase brush strokes and inpaint patches. |
| **Inpaint** | **I** | Toggle Inpaint brush. Paint red semi-transparent regions. On mouse release, the marked area is AI-inpainted and the result is applied as a patch on the stroke canvas (original image preserved). |
| **Clone** | **H** | Toggle Clone brush. Right-click to set source point, then left-click drag to paint from the source onto the stroke canvas. The source moves with the mouse (like Photoshop's Clone Stamp). |
| **Brush** color picker | | Set brush stroke color. |
| **Brush size** slider | | Adjust brush stroke width (1-200 px). Also controls Inpaint and Clone brush size. |

Brush, Eraser, Inpaint, Clone, and Add are mutually exclusive. Activating one deactivates the others.

All strokes (brush, eraser, inpaint patches) live on the same canvas layer. The original image is never modified — inpaint results are rendered as overlay patches that can be erased.

### Navigation

| Button | Shortcut | Action |
|---|---|---|
| **Prev / Next** | Arrow Left / Right | Navigate between pages (disabled in All mode). |

### Display Toggles

| Button | Action |
|---|---|
| **Text Boxes** | Show or hide all text labels and interaction. When off, Save renders only the image + strokes without text boxes. |
| **Outlines** | Show or hide bounding box outlines and corner handles. |
| **Fit to Page** | Toggle between actual-size (scrollable) and fit-to-viewport. |
| **Auto-hide** | Toggle auto-hide toolbar. When on, the toolbar hides and reappears when the mouse nears the top edge. Default is off. |

### View Modes

| Button | Mode |
|---|---|
| **Single** | One page at a time. |
| **Double** | Two pages side-by-side (comic order). |
| **Manga** | Two pages side-by-side (right-to-left). |
| **All** | Vertical webtoon-style scroll of all pages. |

### Text Styling

| Button | Action |
|---|---|
| **A-** / **A+** | Decrease / increase text size. |
| **Aa** | Toggle uppercase text. |
| **LH-** / **LH+** | Decrease / increase line spacing (can go negative). |
| **Font select** | Choose from custom fonts (loaded from `fonts/` folder) or built-in system fonts. |
| **Text** color picker | Set text fill color (next to font selector). |

### Text Outline

| Control | Action |
|---|---|
| **O-** / **O+** | Decrease / increase text outline (stroke) thickness. Starts at 0 (no outline). |
| **OL** color picker | Set text outline color. |

### Background & Color

| Control | Action |
|---|---|
| **BG-** / **BG+** | Decrease / increase text background opacity. |
| **BG** color picker | Set text background color. |

## Mouse Interactions

### Brush / Eraser / Inpaint (when active)

| Action | Effect |
|---|---|
| **Left-click drag** anywhere on image | Paint (Brush), erase (Eraser), mark for inpainting (Inpaint), or clone from source (Clone). |
| **Right-click** on image (Clone tool) | Set the clone source point. |

### Text Labels (when no tool is active, Text Boxes enabled)

| Action | Effect |
|---|---|
| **Left-click drag** on a text label | Move the entire bounding box and its text. |
| **Double left-click** on a text label | Open the text editor to modify the text. |

### Corner Handles (visible when Outlines is on)

| Action | Effect |
|---|---|
| **Left-click drag** a corner | Resize the bounding box. |
| **Right-click drag** a corner | Rotate the bounding box and text. |

### Customize Mode

| Action | Effect |
|---|---|
| **Right-click** on a text label | Enter customize mode for that box (red dashed indicator). Toolbar changes now apply only to this box. |
| **Right-click** on empty space | Exit customize mode. Toolbar changes apply globally again. |
| **Escape** | Exit customize mode. |

Customized text boxes keep their individual settings (font, size, color, outline, etc.) even when global settings change.

### Text Editor

| Key | Action |
|---|---|
| **Enter** | Save text and close editor. If text is empty, the bounding box is deleted. |
| **Shift+Enter** | Insert a newline. |
| **Escape** | Close editor (saves current text; deletes box if empty). |
| Click outside | Save and close. |

## Keyboard Shortcuts

| Key | Action |
|---|---|
| **B** | Toggle Brush tool. |
| **E** | Toggle Eraser tool. |
| **I** | Toggle Inpaint brush tool. |
| **H** | Toggle Clone brush tool. |
| **N** | Toggle Add (new bounding box) mode. |
| **Ctrl+Z** | Undo last change. |
| **Ctrl+Shift+Z** | Redo last undone change. |
| **Arrow Left** | Previous page. |
| **Arrow Right** | Next page. |
| **Escape** | Exit Add mode / deactivate tool / exit customize mode. |

## JSON Import/Export

### Export .json

Saves all bounding boxes as a bundle JSON (`lmt_textbox.json`):

```json
[
  {
    "file": "page1",
    "content": [
      [0.1, 0.2, 0.3, 0.4, "detected text"],
      [0.5, 0.1, 0.6, 0.3, "more text"]
    ],
    "customize": [
      null,
      {"textFont": "'Arial', sans-serif", "textColor": "#ff0000", "textSizeDelta": 4}
    ]
  }
]
```

- File names are base names **without extension**
- Coordinates are normalized to 0-1 range
- `customize` array (optional): per-box style overrides, same length as `content`. `null` for boxes using global settings. Supported keys: `textFont`, `textColor`, `bgColor`, `bgOpacity`, `strokeColor`, `strokeWidth`, `textSizeDelta`, `lineHeightDelta`, `textUppercase`
- Compatible with the Python `manage-extract-and-clean` pipeline

### Import .json

- Supports selecting **multiple JSON files** at once
- Matches entries to loaded images by base name (extension-agnostic)
- Imports `customize` per-box configs if present in the JSON
- Supports both the exported format and the Python pipeline output

### Export script / Import script

- **Export script**: Downloads `lmt_script.txt` with one text line per bounding box from visible images. Newlines within text escaped as `\n`.
- **Import script**: Loads a `.txt` file. Line count must match total visible bbox count. Each line replaces the corresponding box's text. `\n` in the file becomes actual newlines.

## Persistence

Project data is saved to **IndexedDB** automatically (no size limit):

- **Settings** (toolbar state, colors, font, view mode, brush settings) persist in localStorage across sessions.
- **Project data** (loaded images, bounding boxes, text, positions, rotations, brush strokes, inpaint patches) survive page refresh (F5) via IndexedDB.
- Data is saved 0.5 second after the last change (debounced) and immediately on page unload.
- Loading new files via **Load** replaces the saved project data.

## Fonts

Custom fonts are listed in the `CUSTOM_FONTS` array inside `index.html`. To add a new font:

1. Drop the `.ttf`, `.otf`, `.woff`, or `.woff2` file into the `fonts/` folder.
2. Add the filename to the `CUSTOM_FONTS` array in `index.html`.
3. Reload the page.

Custom fonts appear at the top of the font selector dropdown. Five built-in system fonts (System, Arial, Verdana, Georgia, Mono) are always available. Works with `file://` — no web server required.

## AI Models (Extract & Inpaint)

All ML inference runs **client-side** in the browser via [ONNX Runtime Web](https://onnxruntime.ai/). Models are converted from the [manga-image-translator](https://github.com/zyddnys/manga-image-translator) project and loaded lazily on first use.

| Model | Purpose | Size |
|---|---|---|
| `manga_det.onnx` | DBNet ResNet34 text detection (manga-specific) | 306 MB |
| `manga_ocr_encoder.onnx` | ConvNext + Transformer OCR encoder (multilingual) | 103 MB |
| `manga_ocr_decoder.onnx` | Autoregressive OCR decoder (greedy decode) | 105 MB |
| `manga_inpaint.onnx` | AOT-GAN inpainting (manga-trained) | 24 MB |
| `alphabet-all-v7.txt` | OCR character dictionary (46,272 chars) | 183 KB |

**Total: ~538 MB**. Downloaded on first Extract/Inpaint use and cached by the browser.

### How Extract Works

1. **Detection** — DBNet detects text regions in the manga page (letterbox resize to 2048x2048)
2. **OCR** — Each detected textline is cropped (vertical text rotated 90°), resized to h=48, and decoded autoregressively with greedy search + repetition/confidence stopping
3. **Textline merge** — Nearby textlines are grouped into speech bubble regions using font-size-based distance and aspect ratio checks
4. **Mask creation** — Detector's segmentation mask is clipped to detected regions and dilated (configurable via Inpaint pad)
5. **Inpainting** — Each region is processed at 512x512 with AOT-GAN, result applied as patches on the stroke canvas (original image preserved)

### How Inpaint Brush Works

1. User paints red regions with the Inpaint brush
2. On mouse release, a composite of the base image + existing strokes is created
3. Each painted region is cropped with context padding (if smaller than 512x512, the crop is expanded and centered on the mark — no upscaling)
4. AOT-GAN inpaints at 512x512, only masked pixels are kept as a transparent patch
5. Patch is added to the stroke canvas (erasable with the Eraser tool)

### Requirements for AI Features
- **HTTP server required** (e.g. `python -m http.server 8000`) — ONNX models cannot be loaded via `file://`
- **Chrome 113+** or **Edge 113+** recommended (WebGPU backend for faster inference)
- **4GB+ RAM** recommended for inference
- All other features (text editing, brush, save, JSON/script import/export) work without a server

## Technical Details

- Single HTML file + `ml.js` for ML inference. No build step needed.
- Uses HTML5 Canvas for all text rendering, brush strokes, inpaint patches, and overlays.
- Brush strokes and inpaint patches share the same canvas layer (rendered via offscreen canvas with `destination-out` compositing for eraser support).
- Inpaint results are non-destructive overlay patches — the original image is never modified.
- Project data stored in IndexedDB (no size limit, unlike localStorage).
- Supports JPG, JPEG, PNG, GIF, BMP, WebP, TIFF, and AVIF formats.
- Save exports at the image's full natural resolution regardless of display size.
- OCR uses greedy autoregressive decoding with repetition detection and confidence-based early stopping.
- Inpainting uses per-region processing: small regions are centered in 512x512 crops (not upscaled) for maximum quality.
