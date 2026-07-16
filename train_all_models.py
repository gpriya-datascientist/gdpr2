"""
PressVision2 — Multi-model XML annotation training.
Tests llava:7b, qwen2.5vl:7b, minicpm-v against annotated bad images.
Finds best prompt per model per project. Saves results to trained_prompts.json.
"""
import asyncio, base64, json, os, re, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path
import httpx

sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
BASE_DIR   = Path(r"C:\Users\gpngur01\Downloads\gdpr-ai2")

MODELS = ["llava:7b", "qwen2.5vl:7b", "minicpm-v"]

PROJECTS = {
    "BSH_Crack_Detection":   {"part": "BSH Metal Press Tool",        "hint": "Detect dark cracks/fractures on smooth metal surface. Bright glare is NORMAL."},
    "Klinken_Rack_Position": {"part": "Press Clamp Transport Rack",  "hint": "Detect multiple clamps extended per tower. Max 1 clamp extended = GOOD."},
    "Klinken_Loaded_Rack":   {"part": "Press Clamp Transport Rack",  "hint": "Detect too many clamps extended on loaded rack. KS3/KS4 = GOOD."},
    "Klinken_Empty_Rack":    {"part": "Press Clamp Rack (Empty)",    "hint": "Detect all clamps extended on empty rack. All retracted = GOOD."},
}


def load_xml_boxes(xml_path: Path) -> list:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w, h = int(size.find('width').text), int(size.find('height').text)
    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bb = obj.find('bndbox')
        boxes.append({
            'name': name,
            'x_min': round(int(bb.find('xmin').text)/w, 3),
            'y_min': round(int(bb.find('ymin').text)/h, 3),
            'x_max': round(int(bb.find('xmax').text)/w, 3),
            'y_max': round(int(bb.find('ymax').text)/h, 3),
        })
    return boxes


def build_prompt(part: str, hint: str, boxes: list) -> str:
    box_lines = []
    for i, b in enumerate(boxes):
        box_lines.append(
            f"  Region {i+1} ({b['name']}): "
            f"x={int(b['x_min']*100)}-{int(b['x_max']*100)}%, "
            f"y={int(b['y_min']*100)}-{int(b['y_max']*100)}%")
    return f"""You are a strict visual quality inspector for {part}.
Context: {hint}

EXPERT-ANNOTATED DEFECT LOCATIONS:
{chr(10).join(box_lines)}

Carefully inspect EXACTLY these regions of the image.
Confirm whether each annotated defect is visible.

Respond ONLY with valid JSON (no extra text):
{{"verdict": "ANOMALY" or "GOOD", "confidence": 0.0-1.0, "reason": "max 25 words describing what you see", "defect_type": "crack or wrong_position or deformation or unknown", "confirmed": {len(boxes)}}}"""


async def run_model(model: str, img_b64: str, prompt: str, timeout: int = 180) -> dict:
    payload = {"model": model, "prompt": prompt, "images": [img_b64],
               "stream": False, "options": {"temperature": 0.0}}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            t0 = time.perf_counter()
            resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            elapsed = round((time.perf_counter()-t0)*1000)
            raw = resp.json().get("response", "").strip()
        m = re.search(r'\{.*?"verdict".*?\}', raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            return {"verdict": parsed.get("verdict","?").upper(),
                    "confidence": parsed.get("confidence", 0),
                    "reason": parsed.get("reason",""),
                    "elapsed_ms": elapsed, "raw": raw[:200]}
        if "ANOMALY" in raw.upper():
            return {"verdict":"ANOMALY","confidence":0.6,"reason":"detected","elapsed_ms":elapsed}
        return {"verdict":"GOOD","confidence":0.5,"reason":"no defect","elapsed_ms":elapsed}
    except Exception as e:
        return {"verdict":"ERROR","reason":str(e)[:80],"elapsed_ms":0}


async def train_model_on_project(model: str, proj_name: str, proj_cfg: dict) -> dict:
    bad_dir = BASE_DIR / "data" / proj_name / "bad"
    imgs = sorted(bad_dir.glob("*.jpg"))
    annotated = [(img, bad_dir/(img.stem+".xml")) for img in imgs if (bad_dir/(img.stem+".xml")).exists()]

    if not annotated:
        return {"project": proj_name, "model": model, "tested": 0, "accuracy": 0, "note": "no XMLs"}

    correct, total, results = 0, 0, []
    for img_path, xml_path in annotated:
        with open(img_path, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode()
        boxes = load_xml_boxes(xml_path)
        prompt = build_prompt(proj_cfg["part"], proj_cfg["hint"], boxes)
        r = await run_model(model, img_b64, prompt)
        ok = r["verdict"] == "ANOMALY"
        if ok: correct += 1
        total += 1
        status = "OK" if ok else "MISS"
        print(f"    [{status}] {r['verdict']} {r.get('confidence',0):.0%} | {img_path.name[:30]} | {r.get('elapsed_ms',0)}ms")
        results.append({"file": img_path.name, **r, "correct": ok})

    acc = correct/total if total else 0
    return {"project": proj_name, "model": model, "tested": total,
            "correct": correct, "accuracy": round(acc,3), "results": results}


async def main():
    print("PressVision2 — Multi-Model XML Training")
    print("="*60)
    print(f"Models: {MODELS}")
    print(f"Projects: {list(PROJECTS.keys())}\n")

    all_results = {}
    summary = []

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"MODEL: {model}")
        print(f"{'='*60}")
        all_results[model] = {}

        for proj_name, proj_cfg in PROJECTS.items():
            bad_dir = BASE_DIR / "data" / proj_name / "bad"
            xml_count = len(list(bad_dir.glob("*.xml")))
            if xml_count == 0:
                print(f"  SKIP {proj_name} — no XMLs")
                continue
            print(f"\n  Project: {proj_name} ({xml_count} annotated images)")
            result = await train_model_on_project(model, proj_name, proj_cfg)
            all_results[model][proj_name] = result
            acc = result.get("accuracy", 0)
            print(f"  → {result.get('correct',0)}/{result.get('tested',0)} correct ({acc*100:.0f}%)")
            summary.append({"model": model, "project": proj_name,
                            "accuracy": acc, "tested": result.get("tested",0)})

    # Save results
    out = BASE_DIR / "trained_prompts.json"
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Project':<25} {'Accuracy':<10} {'Tested'}")
    print("-"*60)
    for s in summary:
        print(f"{s['model']:<20} {s['project']:<25} {s['accuracy']*100:.0f}%{'':<7} {s['tested']}")

    print(f"\nResults saved to trained_prompts.json")
    print("Use these results to pick best model per project.")


if __name__ == '__main__':
    asyncio.run(main())
