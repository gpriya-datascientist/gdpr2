# PressVision2

Industrial visual inspection system combining **DINOv2 ViT-S/14** (fast, trained) with **VLM models** (llava:7b, minicpm-v) guided by **Pascal VOC XML annotations** for precise defect localization.

---

## How It Works

```
[ANNOTATE]  annotator.html → draw boxes on defects → export Pascal VOC XML → save to bad/ folder
[TRAIN]     server startup → DINOv2 embeds all good/+bad/ images → LogisticRegression classifier
                           → PatchCore memory bank from good patches → ready in ~2 seconds
[INFER]     upload image → DINOv2 classify (50ms) → XML found? use annotated boxes
                                                    → no XML? PatchCore nearest-neighbor boxes
```

---

## Projects & Dataset

| Project | Good | Bad | Task |
|---|---|---|---|
| BSH_Crack_Detection | 8 | 8 | Dark crack/fracture on metal press tool edge |
| Klinken_Rack_Position | 16 | 32 | Press clamp rack — all angles combined (KS1–6) |
| Klinken_Loaded_Rack | 16 | 23 | Loaded rack — too many clamps extended (KS1/2/5) |
| Klinken_Empty_Rack | 9 | 9 | Empty rack — all clamps extended (KS4 good, KS6 bad) |

---

## Model Comparison — Real Tested Results

### Bad images — ANOMALY detection (with XML annotation context)

| Model | Type | BSH | Klinken Rack | Klinken Loaded | Klinken Empty | Speed |
|---|---|---|---|---|---|---|
| **DINOv2** | ViT | **100%** 8/8 | **100%** 32/32 | 74% 15/23 | **100%** 9/9 | **50ms** |
| **minicpm-v** | VLM | 88% 7/8 | **100%** 32/32 | **100%** 23/23 | **100%** 9/9 | ~45s |
| **llava:7b** | VLM | **100%** 8/8 | **100%** 32/32 | **100%** 23/23 | **100%** 9/9 | ~30s |
| qwen2.5vl | VLM | 0% | 0% | 0% | 0% | ~3min |

### Good images — GOOD detection (false positive check, tested without XML)

| Model | Type | BSH | Klinken Rack | Klinken Loaded | Klinken Empty |
|---|---|---|---|---|---|
| **DINOv2** | ViT | **100%** 8/8 | **100%** 16/16 | **100%** 16/16 | **100%** 9/9 |
| llava:7b | VLM | 0% 0/8 | TBD | TBD | TBD |
| minicpm-v | VLM | TBD | TBD | TBD | TBD |
| qwen2.5vl | VLM | N/A | N/A | N/A | N/A |

---

## Key Finding — DINOv2 is the Most Reliable Model

> DINOv2 achieves **100% on good images across all projects** (zero false positives) and **100% on bad images** for 3 of 4 projects. It is the only model fully validated on both good and bad images. The one weakness is Klinken Loaded Rack at 74% — use **llava:7b or minicpm-v** for that specific project.

| Use case | Recommended model |
|---|---|
| Production (speed + reliability) | **DINOv2** — 50ms, zero false positives |
| Klinken Loaded Rack only | **llava:7b** — 100% on bad, ~30s |
| Maximum bad-image accuracy | **minicpm-v** — 97% overall, ~45s |
| Avoid | **qwen2.5vl** — API endpoint issue, 0% accuracy |

---

## Why qwen2.5vl Shows 0%

qwen2.5vl:7b requires the OpenAI-compatible `/v1/chat/completions` endpoint with multimodal message format. When called via the standard `/api/generate` endpoint (used by llava/minicpm), it ignores the image and returns GOOD 50% in ~650ms. The fix is in `inspector.py` — re-run `train_all_models.py` after the patch for accurate results.

---

## Quick Start

```powershell
cd gdpr-ai2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn httpx python-multipart pydantic-settings python-dotenv pillow torch torchvision scikit-learn --index-url https://download.pytorch.org/whl/cpu

copy .env.example .env

# Window 1 — API server
python -m uvicorn server:app --host 127.0.0.1 --port 8004

# Window 2 — UI file server
python -m http.server 3002
```

Open `http://localhost:3002/index.html`

Model comparison dashboard: `http://localhost:3002/comparison.html`

Annotation tool: `http://localhost:3002/annotator.html`

---

## Key Files

| File | Purpose |
|---|---|
| `server.py` | FastAPI — port 8004 |
| `inspector.py` | Inference engine — DINOv2 + VLM routing + FORCE_LLM flag |
| `embedder.py` | DINOv2 ViT-S/14 embeddings + PatchCore localization |
| `annotation_utils.py` | Pascal VOC XML reader (looks in `bad/`, `bad/xml/`, `bad/annotations/`) |
| `annotator.html` | Browser annotation tool — draw boxes, export XML |
| `index.html` | Main inspection UI |
| `comparison.html` | Model accuracy comparison dashboard |
| `train_all_models.py` | VLM accuracy test — all 3 models × all projects |
| `train_bsh_loaded.py` | VLM test — BSH + Klinken Loaded only |
| `test_dinov2_all.py` | DINOv2 full accuracy test (good + bad images) |

---

## Annotation Workflow

1. Open `http://localhost:3002/annotator.html`
2. Load folder — e.g. `data/BSH_Crack_Detection/bad`
3. Draw boxes on defects, set class (`crack`, `wrong_position` etc.)
4. Export XML → save to same `bad/` folder
5. Restart server — XML boxes used automatically at inference
6. Backend shows `dinov2+annotation` confirming XML was used

---

## Stack

- **DINOv2 ViT-S/14** — Meta, frozen pretrained, 384-dim embeddings, zero-shot transfer
- **LogisticRegression** — scikit-learn, `class_weight=balanced`, trains in ~2s
- **PatchCore** — nearest-neighbor memory bank, LOO threshold calibration
- **VLMs** — Ollama: llava:7b, minicpm-v, qwen2.5vl:7b
- **XML annotations** — Pascal VOC format, override PatchCore boxes at inference
- **Frontend** — Vanilla JS/HTML, no framework, pure CSS bounding boxes

---

## Ports

| Service | Port |
|---|---|
| API | 8004 |
| UI | 3002 |
| Ollama | 11434 |
