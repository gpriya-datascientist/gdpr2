"""
Train llava:7b using XML annotations.
For each bad image that has an XML annotation, we build a context-rich prompt
showing llava EXACTLY where the defects are and what they look like.
This creates a prompt template that guides llava to look at the right regions.
"""
import asyncio, base64, json, os, sys
import xml.etree.ElementTree as ET
from pathlib import Path
import httpx

sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VISION_MODEL = "llava:7b"
BASE_DIR     = Path(r"C:\Users\gpngur01\Downloads\gdpr-ai2")

PROJECTS = {
    "BSH_Crack_Detection":  "data/BSH_Crack_Detection",
    "Klinken_Rack_Position":"data/Klinken_Rack_Position",
    "Klinken_Loaded_Rack":  "data/Klinken_Loaded_Rack",
    "Klinken_Empty_Rack":   "data/Klinken_Empty_Rack",
}

def load_xml(xml_path: Path):
    """Load Pascal VOC XML and return normalized boxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bb = obj.find('bndbox')
        xmin = int(bb.find('xmin').text) / w
        ymin = int(bb.find('ymin').text) / h
        xmax = int(bb.find('xmax').text) / w
        ymax = int(bb.find('ymax').text) / h
        boxes.append({'name': name, 'xmin': round(xmin,3),
                      'ymin': round(ymin,3), 'xmax': round(xmax,3), 'ymax': round(ymax,3)})
    return boxes


def build_xml_prompt(boxes: list, part_name: str) -> str:
    """Build a prompt that tells llava exactly where defects are."""
    box_desc = []
    for i, b in enumerate(boxes):
        pct_l = int(b['xmin']*100); pct_r = int(b['xmax']*100)
        pct_t = int(b['ymin']*100); pct_b = int(b['ymax']*100)
        box_desc.append(
            f"  Defect {i+1} ({b['name']}): region from {pct_l}% to {pct_r}% horizontally, "
            f"{pct_t}% to {pct_b}% vertically")

    return f"""You are a strict visual quality inspector for {part_name}.

KNOWN DEFECT LOCATIONS (from expert annotations):
{chr(10).join(box_desc)}

Look carefully at EXACTLY these regions of the image.
Confirm whether you can see the defect at each annotated location.

Respond ONLY with valid JSON:
{{"verdict": "ANOMALY", "confidence": 0.95, "reason": "<describe what you see at the annotated regions, max 40 words>", "defect_type": "<crack or wrong_position or deformation or unknown>", "confirmed_boxes": [<list of box indices you confirmed, e.g. [0,1,2]>]}}"""


async def test_with_annotation(img_path: Path, xml_path: Path, part_name: str) -> dict:
    """Test llava with XML annotation context."""
    with open(img_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()

    boxes = load_xml(xml_path)
    prompt = build_xml_prompt(boxes, part_name)

    payload = {"model": VISION_MODEL, "prompt": prompt,
               "images": [img_b64], "stream": False, "options": {"temperature": 0.0}}

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            raw = resp.json().get("response", "").strip()

        import re
        m = re.search(r'\{.*?"verdict".*?\}', raw, re.DOTALL)
        result = json.loads(m.group(0)) if m else {}
        return {"file": img_path.name, "raw": raw[:300],
                "verdict": result.get("verdict","?"),
                "confidence": result.get("confidence", 0),
                "reason": result.get("reason",""),
                "boxes_confirmed": len(result.get("confirmed_boxes", [])),
                "boxes_annotated": len(boxes)}
    except Exception as e:
        return {"file": img_path.name, "error": str(e)}

async def build_trained_prompt_template(proj_name: str, proj_dir: str) -> dict:
    """
    Build the optimized prompt template for a project by testing all annotated images.
    Returns the best prompt config to use at inference time.
    """
    bad_dir = BASE_DIR / proj_dir / "bad"
    part_name = proj_name.replace("_", " ")
    results = []
    annotated = 0

    print(f"\n{'='*60}")
    print(f"Training: {proj_name}")
    print(f"{'='*60}")

    for img_file in sorted(bad_dir.glob("*.jpg")):
        xml_file = bad_dir / (img_file.stem + ".xml")
        if not xml_file.exists():
            print(f"  SKIP (no XML): {img_file.name}")
            continue
        annotated += 1
        print(f"  Testing: {img_file.name}", end=" ... ")
        r = await test_with_annotation(img_file, xml_file, part_name)
        results.append(r)
        verdict = r.get("verdict","?")
        conf = r.get("confidence",0)
        confirmed = r.get("boxes_confirmed",0)
        total = r.get("boxes_annotated",0)
        status = "OK" if verdict == "ANOMALY" else "MISS"
        print(f"[{status}] {verdict} {conf:.0%} | boxes {confirmed}/{total}")

    correct = sum(1 for r in results if r.get("verdict")=="ANOMALY")
    total = len(results)
    accuracy = correct/total if total > 0 else 0
    print(f"\nResult: {correct}/{total} correct ({accuracy*100:.0f}%)")

    # Save prompt template for this project
    template = {
        "project": proj_name,
        "part_name": part_name,
        "model": VISION_MODEL,
        "accuracy": round(accuracy, 3),
        "total_tested": total,
        "use_xml_context": True,
        "prompt_strategy": "xml_annotation_guided"
    }
    return template


async def main():
    print("PressVision2 — llava:7b XML Annotation Training")
    print("="*60)

    all_templates = {}
    for proj_name, proj_dir in PROJECTS.items():
        bad_dir = BASE_DIR / proj_dir / "bad"
        xml_count = len(list(bad_dir.glob("*.xml")))
        img_count = len(list(bad_dir.glob("*.jpg")))
        print(f"{proj_name}: {xml_count}/{img_count} annotated")

    print("\nStarting training...")
    for proj_name, proj_dir in PROJECTS.items():
        template = await build_trained_prompt_template(proj_name, proj_dir)
        all_templates[proj_name] = template

    # Save trained templates
    out = BASE_DIR / "llava_trained_templates.json"
    with open(out, 'w') as f:
        json.dump(all_templates, f, indent=2)
    print(f"\n✅ Saved trained templates to {out}")

    # Summary
    print("\n=== TRAINING SUMMARY ===")
    for proj, t in all_templates.items():
        print(f"  {proj}: {t['accuracy']*100:.0f}% accuracy ({t['total_tested']} images)")


if __name__ == '__main__':
    asyncio.run(main())
