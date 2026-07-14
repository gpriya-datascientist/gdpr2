"""
PressVision2 Inspector
Primary: DINOv2 + logistic regression (~50ms, trained on your images)
Fallback: llava:7b via Ollama (~30s)
"""
import base64, json, logging, time, os, asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import httpx
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava:7b")
GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

BASE_DIR = Path(__file__).parent

PROJECTS = {
    "bsh": {
        "name": "BSH Crack Detection",
        "description": "BSH metal press tool edge — detect cracks",
        "part_name": "BSH Metal Press Tool",
        "description_hint": "Dark irregular cracks breaking the smooth metal surface are ANOMALY. Bright glare and reflections are GOOD.",
        "good_dir": "data/bsh/good",
        "bad_dir":  "data/bsh/bad",
        "control_param":  "left_door_stroke_rate",
        "control_action": "reduce",
        "control_pct":    15,
    },
    "klinken": {
        "name": "Klinken-Rack Position",
        "description": "Press clamp rack — detect incorrect clamp positions",
        "part_name": "Press Clamp Transport Rack",
        "description_hint": "GOOD: max 1 clamp extended per tower or all retracted. ANOMALY: multiple clamps extended per tower.",
        "good_dir": "data/klinken/good",
        "bad_dir":  "data/klinken/bad",
        "control_param":  "cycle_time_s",
        "control_action": "increase",
        "control_pct":    10,
    },
    "casting": {
        "name": "Casting Defect Detection",
        "description": "Industrial casting impeller — detect surface defects",
        "part_name": "Cast Impeller Part",
        "description_hint": "GOOD: smooth uniform casting surface. ANOMALY: surface defects, pits, cracks, or deformations on the casting.",
        "good_dir": "data/casting/good",
        "bad_dir":  "data/casting/bad",
        "control_param":  "assembly_speed_ms",
        "control_action": "reduce",
        "control_pct":    10,
    },
}

# ── DINOv2 models (trained on startup) ───────────────────────────────────
_embedder = None
_models: dict[str, dict] = {}

def reset_models():
    """Call this to force retrain all models."""
    global _models
    _models = {}


def _get_embedder():
    global _embedder
    if _embedder is None:
        from embedder import DinoV2Embedder
        _embedder = DinoV2Embedder()
    return _embedder


def _get_model(project_id: str):
    if project_id not in _models:
        proj = PROJECTS[project_id]
        good_dir = str(BASE_DIR / proj["good_dir"])
        bad_dir  = str(BASE_DIR / proj["bad_dir"])
        good_count = len([f for f in Path(good_dir).iterdir() if f.suffix.lower() in {".jpg",".jpeg",".png"}]) if Path(good_dir).exists() else 0
        bad_count  = len([f for f in Path(bad_dir).iterdir()  if f.suffix.lower() in {".jpg",".jpeg",".png"}]) if Path(bad_dir).exists() else 0
        if good_count >= 2 and bad_count >= 2:
            from embedder import train_classifier
            log.info("Training DINOv2 model for project '%s' (%d good, %d bad)...", project_id, good_count, bad_count)
            _models[project_id] = train_classifier(_get_embedder(), good_dir, bad_dir)
            log.info("DINOv2 model ready for '%s'", project_id)
        else:
            log.warning("Not enough images for '%s' (need >=2 good, >=2 bad)", project_id)
            _models[project_id] = None
    return _models[project_id]


@dataclass
class InspectionResult:
    verdict: str
    confidence: float
    reason: str
    defect_type: Optional[str]
    backend_used: str
    latency_ms: float
    project: str = ""
    boxes: list = field(default_factory=list)
    control_suggestion: Optional[dict] = None
    raw_response: str = ""


async def inspect_image(image_bytes: bytes, project_id: str = "bsh",
                        mime: str = "image/jpeg") -> InspectionResult:
    proj = PROJECTS.get(project_id, PROJECTS["bsh"])
    start = time.perf_counter()

    # ── Primary: DINOv2 fast local classifier (~50ms) ─────────────────────
    try:
        model = await asyncio.to_thread(_get_model, project_id)
        if model is not None:
            from embedder import predict
            verdict, conf, boxes, reason = await asyncio.to_thread(
                predict, model, _get_embedder(), image_bytes)
            latency = (time.perf_counter()-start)*1000
            r = InspectionResult(verdict=verdict, confidence=conf, reason=reason,
                defect_type="crack" if verdict=="ANOMALY" else None,
                backend_used="dinov2", latency_ms=latency,
                project=project_id, boxes=boxes)
            r.control_suggestion = _get_suggestion(r, proj)
            return r
    except Exception as e:
        log.warning("DINOv2 failed: %s", e)

    # ── Fallback: llava via Ollama ─────────────────────────────────────────
    r = await _try_local(image_bytes, proj, mime)
    if r:
        r.project = project_id
        r.control_suggestion = _get_suggestion(r, proj)
        return r

    return InspectionResult(verdict="ERROR", confidence=0.0, project=project_id,
        reason="All backends failed.", defect_type=None,
        backend_used="none", latency_ms=(time.perf_counter()-start)*1000)


PROMPT_TEMPLATE = """You are a strict visual quality inspector in an industrial press shop.
Inspection object: {part_name}.
{description_hint}
Respond ONLY with valid JSON:
{{"verdict": "ANOMALY" or "GOOD", "confidence": <0.0-1.0>, "reason": "<max 30 words>", "defect_type": "<crack or wrong_position or unknown or null>", "box": {{"x_min": 0.1, "y_min": 0.3, "x_max": 0.7, "y_max": 0.8}} or null}}"""


async def _try_local(image_bytes, proj, mime):
    start = time.perf_counter()
    try:
        img_b64 = base64.b64encode(image_bytes).decode()
        prompt = PROMPT_TEMPLATE.format(
            part_name=proj["part_name"],
            description_hint=proj["description_hint"])
        payload = {"model": VISION_MODEL, "prompt": prompt,
                   "images": [img_b64], "stream": False,
                   "options": {"temperature": 0.0}}
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            resp.raise_for_status()
            raw = resp.json().get("response","").strip()
        return _parse(raw, VISION_MODEL, (time.perf_counter()-start)*1000)
    except Exception as e:
        log.warning("llava failed: %s", e)
        return None


def _parse(raw, backend, latency_ms):
    import re
    try:
        clean = raw.strip()
        if "```" in clean:
            for p in clean.split("```"):
                p = p.strip()
                if p.startswith("json"): p = p[4:].strip()
                if '"verdict"' in p: clean = p; break
        m = re.search(r'\{.*?"verdict".*?\}', clean, re.DOTALL)
        if m: clean = m.group(0)
        parsed = json.loads(clean)
        v = str(parsed.get("verdict","GOOD")).upper()
        if v not in ("GOOD","ANOMALY"): v = "GOOD"
        conf = max(0.0, min(1.0, float(parsed.get("confidence",0.5))))
        reason = str(parsed.get("reason",""))[:300]
        dt = parsed.get("defect_type")
        if isinstance(dt,list): dt = dt[0] if dt else None
        if dt in ("null","",None): dt = None
        if dt: dt = str(dt).lower()
        boxes = []
        box = parsed.get("box")
        if box and isinstance(box,dict) and v=="ANOMALY":
            try: boxes=[{"x_min":float(box.get("x_min",0.1)),"y_min":float(box.get("y_min",0.3)),
                         "x_max":float(box.get("x_max",0.7)),"y_max":float(box.get("y_max",0.8))}]
            except: pass
        return InspectionResult(verdict=v, confidence=conf, reason=reason,
            defect_type=dt, backend_used=backend, latency_ms=latency_ms, boxes=boxes)
    except Exception as e:
        log.warning("Parse failed: %s | raw: %s", e, raw[:150])
        if "ANOMALY" in raw.upper():
            return InspectionResult(verdict="ANOMALY", confidence=0.6,
                reason="Defect detected", defect_type="unknown",
                backend_used=backend, latency_ms=latency_ms)
        return InspectionResult(verdict="GOOD", confidence=0.5,
            reason="No defect", defect_type=None,
            backend_used=backend, latency_ms=latency_ms)


def _get_suggestion(result, proj):
    if result.verdict != "ANOMALY": return None
    param, action, pct = proj["control_param"], proj["control_action"], proj["control_pct"]
    bases = {"left_door_stroke_rate":45.0,"right_door_stroke_rate":44.0,
             "hood_stroke_rate":30.0,"assembly_speed_ms":2.8,"cycle_time_s":240.0,"torque_nm":150.0}
    base = bases.get(param,1.0)
    mult = (1-pct/100) if action=="reduce" else (1+pct/100)
    label = param.replace("_"," ")
    return {"param":param,"param_label":label,"action":action,"pct":pct,
            "current_value":base,"suggested_value":round(base*mult,2),
            "change":f"{'-' if action=='reduce' else '+'}{pct}%",
            "command":f"{'Reduce' if action=='reduce' else 'Increase'} {label} by {pct}%",
            "risk":"medium" if pct>12 else "low"}
