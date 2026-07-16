"""
Train llava:7b, qwen2.5vl:7b, minicpm-v on:
- BSH_Crack_Detection/bad
- Klinken_Loaded_Rack/bad
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
MODELS     = ["llava:7b", "qwen2.5vl:7b", "minicpm-v"]

PROJECTS = {
    "BSH_Crack_Detection": {
        "part": "BSH Metal Press Tool",
        "hint": "Detect dark irregular cracks or fractures on smooth metal surface. Bright glare is NORMAL."
    },
    "Klinken_Loaded_Rack": {
        "part": "Press Clamp Transport Rack (Loaded)",
        "hint": "Detect multiple clamps extended per tower on loaded rack. Max 1 clamp extended = GOOD."
    },
}


def load_xml_boxes(xml_path):
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


def build_prompt(part, hint, boxes):
    box_lines = [
        f"  Region {i+1} ({b['name']}): "
        f"x={int(b['x_min']*100)}-{int(b['x_max']*100)}%, "
        f"y={int(b['y_min']*100)}-{int(b['y_max']*100)}%"
        for i, b in enumerate(boxes)
    ]
    return f"""You are a strict visual quality inspector for {part}.
Context: {hint}

EXPERT-ANNOTATED DEFECT LOCATIONS:
{chr(10).join(box_lines)}

Inspect EXACTLY these regions. Confirm if defect is visible there.

Respond ONLY with valid JSON:
{{"verdict": "ANOMALY" or "GOOD", "confidence": 0.0-1.0, "reason": "max 25 words", "defect_type": "crack or wrong_position or deformation or unknown"}}"""


async def run_model(model, img_b64, prompt, mime="image/jpeg"):
    # qwen needs OpenAI-compatible endpoint
    if "qwen" in model.lower():
        payload = {"model": model, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
            {"type": "text", "text": prompt}
        ]}], "stream": False, "temperature": 0.0}
        endpoint = f"{OLLAMA_URL}/v1/chat/completions"
    else:
        payload = {"model": model, "prompt": prompt, "images": [img_b64],
                   "stream": False, "options": {"temperature": 0.0}}
        endpoint = f"{OLLAMA_URL}/api/generate"
    try:
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(endpoint, json=payload)
            elapsed = round((time.perf_counter()-t0)*1000)
        if "qwen" in model.lower():
            raw = resp.json()["choices"][0]["message"]["content"].strip()
        else:
            raw = resp.json().get("response", "").strip()
        m = re.search(r'\{.*?"verdict".*?\}', raw, re.DOTALL)
        if m:
            p = json.loads(m.group(0))
            return {"verdict": p.get("verdict","?").upper(),
                    "confidence": p.get("confidence", 0),
                    "reason": p.get("reason",""),
                    "elapsed_ms": elapsed}
        if "ANOMALY" in raw.upper():
            return {"verdict": "ANOMALY", "confidence": 0.6, "elapsed_ms": elapsed}
        return {"verdict": "GOOD", "confidence": 0.5, "elapsed_ms": elapsed}
    except Exception as e:
        return {"verdict": "ERROR", "reason": str(e)[:60], "elapsed_ms": 0}


async def train_project(model, proj_name, proj_cfg):
    bad_dir = BASE_DIR / "data" / proj_name / "bad"
    annotated = [(img, bad_dir/(img.stem+".xml"))
                 for img in sorted(bad_dir.glob("*.jpg"))
                 if (bad_dir/(img.stem+".xml")).exists()]
    if not annotated:
        print(f"  SKIP — no XMLs in {proj_name}")
        return None
    correct, total = 0, 0
    for img_path, xml_path in annotated:
        with open(img_path,'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode()
        boxes  = load_xml_boxes(xml_path)
        prompt = build_prompt(proj_cfg["part"], proj_cfg["hint"], boxes)
        r      = await run_model(model, img_b64, prompt)
        ok     = r["verdict"] == "ANOMALY"
        if ok: correct += 1
        total += 1
        status = "OK" if ok else "MISS"
        print(f"    [{status}] {r['verdict']} {r.get('confidence',0):.0%} | "
              f"{img_path.name[:35]} | {r.get('elapsed_ms',0)}ms")
    acc = correct/total if total else 0
    print(f"  → {correct}/{total} correct ({acc*100:.0f}%)\n")
    return {"project": proj_name, "model": model,
            "correct": correct, "total": total, "accuracy": round(acc,3)}


async def main():
    print("PressVision2 — Training BSH + Klinken_Loaded_Rack")
    print("="*60)
    # Check XMLs
    for proj in PROJECTS:
        bad = BASE_DIR/"data"/proj/"bad"
        xmls = len(list(bad.glob("*.xml")))
        imgs = len(list(bad.glob("*.jpg")))
        print(f"{proj}: {xmls}/{imgs} annotated")
    print()

    all_results = {}
    summary = []
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"MODEL: {model}")
        print(f"{'='*60}")
        for proj_name, proj_cfg in PROJECTS.items():
            bad = BASE_DIR/"data"/proj_name/"bad"
            if not list(bad.glob("*.xml")):
                print(f"  SKIP {proj_name} — annotate images first!")
                continue
            print(f"\n  [{proj_name}]")
            r = await train_project(model, proj_name, proj_cfg)
            if r:
                all_results[f"{model}_{proj_name}"] = r
                summary.append(r)

    # Save
    out = BASE_DIR/"trained_prompts_bsh_loaded.json"
    with open(out,'w') as f: json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Project':<28} {'Accuracy':<10} {'Correct'}")
    print("-"*65)
    for s in summary:
        print(f"{s['model']:<20} {s['project']:<28} "
              f"{s['accuracy']*100:.0f}%{'':<7} {s['correct']}/{s['total']}")
    print(f"\nSaved to trained_prompts_bsh_loaded.json")


if __name__ == '__main__':
    asyncio.run(main())
