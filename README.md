# PressVision2 — Industrial Visual Inspection

Fast industrial defect detection using DINOv2 + logistic regression (~50ms/image).

## Projects
- **BSH Crack Detection** — Metal press tool edge crack detection (8+8 images)
- **Klinken-Rack Position** — Press clamp rack position check (16+32 images)  
- **Casting Defect Detection** — Cast impeller surface defects (40+40 images)

## Architecture
- **Primary**: DINOv2 ViT-S/14 embeddings + logistic regression (~50ms, no cloud)
- **Fallback**: llava:7b via Ollama (~30s)
- **Bounding boxes**: PatchCore-style patch memory bank localization

## Quick Start
```powershell
cd gdpr-ai2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn httpx python-multipart pydantic-settings python-dotenv pillow torch torchvision scikit-learn --index-url https://download.pytorch.org/whl/cpu

# Copy .env.example to .env and fill in keys
copy .env.example .env

# Start API
python -m uvicorn server:app --host 127.0.0.1 --port 8004

# Start UI (separate terminal)
python -m http.server 3002
```

Open http://localhost:3002/index.html

## Dataset Setup
```powershell
# Download Kaggle casting dataset (optional, needs kaggle token)
python download_datasets.py

# Convert all images to grayscale for consistent training
python convert_grayscale.py
```

## Stack
- FastAPI + DINOv2 + scikit-learn
- Ollama (llava:7b / qwen2.5vl:7b) as fallback
- Gemini 2.0 Flash as cloud fallback
