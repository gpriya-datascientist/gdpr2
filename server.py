"""PressVision2 FastAPI Server — Port 8004"""
import asyncio, logging, os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from inspector import inspect_image, PROJECTS

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Base directory — always relative to this file
BASE_DIR = Path(__file__).parent

app = FastAPI(title="PressVision2", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"])


class InspectResponse(BaseModel):
    verdict: str
    confidence: float
    reason: str
    defect_type: Optional[str]
    backend_used: str
    latency_ms: float
    project: str
    boxes: list
    control_suggestion: Optional[dict]


@app.get("/health")
async def health():
    return {"status": "ok", "service": "PressVision2",
            "projects": list(PROJECTS.keys()),
            "vision_model": os.getenv("OLLAMA_VISION_MODEL")}


@app.post("/inspect", response_model=InspectResponse)
async def inspect(
    file: UploadFile = File(...),
    project: str = Form(default="bsh"),
    model: str = Form(default=""),
):
    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large")
    mime = file.content_type or "image/jpeg"
    if not mime.startswith("image/"):
        raise HTTPException(status_code=400, detail="Must be an image")
    # Override model if specified
    if model and model != 'dinov2':
        import inspector as insp
        insp.VISION_MODEL = model
    # Build image_path so annotation_utils can find matching XML
    proj = inspect_image.__module__ and __import__('inspector').PROJECTS.get(project)
    img_path = None
    if proj:
        from pathlib import Path
        bad_dir = Path(__file__).parent / proj.get('bad_dir','')
        candidate = bad_dir / file.filename
        if candidate.exists():
            img_path = str(candidate)
    result = await inspect_image(data, project_id=project, mime=mime, image_path=img_path)
    return InspectResponse(
        verdict=result.verdict, confidence=result.confidence,
        reason=result.reason, defect_type=result.defect_type,
        backend_used=result.backend_used, latency_ms=round(result.latency_ms,1),
        project=result.project, boxes=result.boxes,
        control_suggestion=result.control_suggestion,
    )


@app.get("/projects")
async def get_projects():
    return {"projects": [{"id": k, "name": v["name"],
                          "description": v["description"]}
                         for k,v in PROJECTS.items()]}


@app.get("/dataset/{project_id}/stats")
async def dataset_stats(project_id: str):
    proj = PROJECTS.get(project_id)
    if not proj: raise HTTPException(404, "Project not found")
    good_dir = BASE_DIR / proj["good_dir"]
    bad_dir  = BASE_DIR / proj["bad_dir"]
    good = len([f for f in os.listdir(good_dir)
                if f.lower().endswith(('.jpg','.jpeg','.png'))]) if good_dir.exists() else 0
    bad  = len([f for f in os.listdir(bad_dir)
                if f.lower().endswith(('.jpg','.jpeg','.png'))]) if bad_dir.exists() else 0
    return {"project": project_id, "good_images": good,
            "bad_images": bad, "total": good+bad}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8004, reload=False)
