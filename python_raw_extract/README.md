# Manga Extract & Clean

A GUI application that extracts text from manga images using OCR, removes text from images via inpainting, and manages JSON files for the translation workflow.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for reasonable processing speed)

## Installation

```bash
pip install -r requirements.txt
```

For PyTorch with CUDA, you may need to install from the [PyTorch website](https://pytorch.org/get-started/locally/) first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Usage

```bash
python app.py
```

### Workflow

#### Step 1: Process Manga Images

1. Click **Browse** and select a folder containing manga images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`)
2. Click **Process**
3. The app will:
   - Detect and OCR text in each image -> saves `output/{image_name}.json` (array of `[x0, y0, x1, y1, "text"]`)
   - Inpaint/remove text from each image -> saves `output/{image_name}.{ext}`
   - Combine all JSON files -> `output/lmt_raw.json`
   - Extract text to plain script -> `output/lmt_script.txt`
4. Click **Open Folder** to view the output

#### Step 2: Translate (External)

Take `output/lmt_script.txt` and feed it to an LLM for translation. The file contains one text string per line (with `\n` escaping for newlines). Get back a translated text file with the same structure (one line per translation).

#### Step 3: Turn Translated Script to JSON

1. Select the translated script file using **Browse**
2. Click **Convert to JSON**
3. The app merges translations back into the original structure -> `output/lmt_{translated_name}_converted.json`

## Output Files

| File | Description |
|------|-------------|
| `output/{image}.json` | Per-image text regions: `[x0, y0, x1, y1, "text"]` |
| `output/{image}.{ext}` | Cleaned image with text removed |
| `output/lmt_raw.json` | Combined JSON with all text regions |
| `output/lmt_script.txt` | Plain text script (one line per text, for translation) |
| `output/lmt_*_converted.json` | Translated JSON (full structure restored) |

## Project Structure

```
app.py              # Entry point
gui.py              # Tkinter GUI
backend.py          # Processing logic (pipeline orchestration, JSON ops)
manga_translator/   # Vendored ML pipeline (detection, OCR, inpainting)
  detection/        # Text detection (DBNet)
  ocr/              # Optical character recognition (48px model)
  inpainting/       # Text removal (AOT/LaMa inpainting)
  textline_merge/   # Merges detected text lines into regions
  mask_refinement/  # Refines text masks for cleaner inpainting
  utils/            # Shared utilities, data classes, model loading
  config.py         # Configuration enums and defaults
```

ML models are automatically downloaded on first run to `~/.manga-image-translator/models/`.
