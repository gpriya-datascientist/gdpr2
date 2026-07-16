# PressVision2

Fast industrial defect detection using **DINOv2 ViT-S/14** + **VLM models** (llava:7b, minicpm-v) with XML annotation-guided localization.

## Architecture

```
[ANNOTATE]  annotator.html → draw boxes on bad/ images → export Pascal VOC XML
[TRAIN]     server startup → DINOv2 embeds good/+bad/ → LogisticRegression + PatchCore memory bank → ~2s
[INFER]     upload image → DINOv2 classify (50ms) → XML exists? use annotated boxes : PatchCore boxes
```

## Projects

| Project | Good | Bad | Description |
|---|---|---|---|
| BSH_Crack_Detection | 8 | 8 | Metal press tool edge crack detection |
| Klinken_Rack_Position | 16 | 32 | Press clamp rack — all angles (KS1-6) |
| Klinken_Loaded_Rack | 16 | 23 | Loaded rack clamp position (KS1/2/5) |
| Klinken_Empty_Rack | 9 | 9 | Empty rack clamp position (KS4/6) |

## Model Accuracy (tested on bad images with XML annotations)

| Model | Type | BSH | Klinken Rack | Klinken Loaded | Klinken Empty | Speed |
|---|---|---|---|---|---|---|
| **DINOv2** | ViT | 100% | 100% | 74% | 100% | 50ms |
| **minicpm-v** | VLM | 88% | 100% | 100% | 100% | ~45s |
| **llava:7b** | VLM | 100% | 100% | 100% | 100% | ~30s |
| qwen2.5vl | VLM | 0% | 0% | 0% | 0% | ~3min |

> DINOv2 is default (fastest). minicpm-v best for Klinken Loaded Rack. llava:7b best balance of speed + accuracy.

## Quick Start

```powershell
cd gdpr-ai2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn httpx python-multipart pydantic-settings python-dotenv pillow torch torchvision scikit-learn --index-url https://download.pytorch.org/whl/cpu

copy .env.example .env

# Window 1 — API
python -m uvicorn server:app --host 127.0.0.1 --port 8004

# Window 2 — UI
python -m http.server 3002
```

Open `http://localhost:3002/index.html`

## Key Files

| File | Purpose |
|---|---|
| `server.py` | FastAPI server — port 8004 |
| `inspector.py` | Inference engine — DINOv2 + llava/minicpm routing |
| `embedder.py` | DINOv2 ViT-S/14 embeddings + PatchCore localization |
| `annotation_utils.py` | Pascal VOC XML reader/writer |
| `annotator.html` | Browser-based annotation tool |
| `index.html` | Main inspection UI |
| `comparison.html` | Model accuracy comparison dashboard |
| `train_all_models.py` | VLM training script (all projects) |
| `train_bsh_loaded.py` | VLM training for BSH + Klinken Loaded |

## Annotation Workflow

1. Open `http://localhost:3002/annotator.html`
2. Load folder (e.g. `data/BSH_Crack_Detection/bad`)
3. Draw boxes on defects, set class name (`crack`, `wrong_position` etc.)
4. Export XML → save to same `bad/` folder
5. Restart server — XML boxes used automatically at inference

## Stack

- **Backend**: FastAPI + DINOv2 (Meta) + scikit-learn
- **VLM**: Ollama (llava:7b, minicpm-v, qwen2.5vl:7b)
- **Localization**: PatchCore nearest-neighbor + Pascal VOC XML override
- **Frontend**: Vanilla JS/HTML — no framework

## Dataset Setup

```powershell
# Download Kaggle casting dataset (optional)
python download_datasets.py

# Convert all images to grayscale
python convert_grayscale.py

# Extract klinken images from zip
python extract_klinken_all.py
```

## Ports

| Service | Port | URL |
|---|---|---|
| API | 8004 | http://localhost:8004 |
| UI | 3002 | http://localhost:3002 |
| Ollama | 11434 | http://localhost:11434 |
