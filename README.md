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

## Model Accuracy — Bad Images (ANOMALY detection, tested with XML annotations)

| Model | Type | BSH | Klinken Rack | Klinken Loaded | Klinken Empty | Speed |
|---|---|---|---|---|---|---|
| **DINOv2** | ViT | 100% (8/8) | 100% (32/32) | 74% (15/23) | 100% (9/9) | **50ms** |
| **minicpm-v** | VLM | 88% (7/8) | 100% (32/32) | 100% (23/23) | 100% (9/9) | ~45s |
| **llava:7b** | VLM | 100% (8/8) | 100% (32/32) | 100% (23/23) | 100% (9/9) | ~30s |
| qwen2.5vl | VLM | 0% | 0% | 0% | 0% | ~3min |

## Model Accuracy — Good Images (GOOD detection, false positive rate)

| Model | Type | BSH | Klinken Rack | Notes |
|---|---|---|---|---|
| **DINOv2** | ViT | 100% (8/8) | 100% (16/16) | No false positives |
| **minicpm-v** | VLM | TBD | TBD | Good at following context |
| **llava:7b** | VLM | 0% (0/8) | TBD | High false positive on BSH good images |
| qwen2.5vl | VLM | N/A | N/A | API broken — see note below |

> **Recommendation:** DINOv2 for production (fastest, no false positives). minicpm-v for Klinken Loaded Rack (100% vs DINOv2's 74%). Avoid llava:7b for good image classification.

## Why qwen2.5vl shows 0% / API error

qwen2.5vl:7b requires the **OpenAI-compatible `/v1/chat/completions`** endpoint with multimodal message format, NOT the standard `/api/generate` endpoint that llava/minicpm use. When called via `/api/generate`, it responds in ~650ms with GOOD 50% confidence — meaning it ignores the image entirely and returns a default response. The fix is in `inspector.py` (`patch_qwen.py`) but training results still show 0% because the training script was run before the fix was applied. Re-running `train_all_models.py` after the patch will give accurate qwen results.


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
| `train_all_models.py` | VLM training script — all projects |
| `train_bsh_loaded.py` | VLM training — BSH + Klinken Loaded |
| `test_dinov2_all.py` | DINOv2 accuracy test — all projects |

## Annotation Workflow

1. Open `http://localhost:3002/annotator.html`
2. Load folder (e.g. `data/BSH_Crack_Detection/bad`)
3. Draw boxes on defects, set class name (`crack`, `wrong_position` etc.)
4. Export XML → save to same `bad/` folder
5. Restart server — XML boxes used automatically at inference

## Stack

- **Backend**: FastAPI + DINOv2 (Meta ViT-S/14) + scikit-learn
- **VLM**: Ollama (llava:7b, minicpm-v, qwen2.5vl:7b)
- **Localization**: PatchCore nearest-neighbor + Pascal VOC XML override
- **Frontend**: Vanilla JS/HTML — no framework

## Ports

| Service | Port | URL |
|---|---|---|
| API | 8004 | http://localhost:8004 |
| UI | 3002 | http://localhost:3002 |
| Ollama | 11434 | http://localhost:11434 |
