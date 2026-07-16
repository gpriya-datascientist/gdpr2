"""Test all models on ALL good images — check false positive rate."""
import asyncio, base64, json, re, sys, time, os
import xml.etree.ElementTree as ET
from pathlib import Path
import httpx
sys.path.insert(0,'.')
from dotenv import load_dotenv
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL","http://localhost:11434")
BASE_DIR   = Path(r"C:\Users\gpngur01\Downloads\gdpr-ai2")
MODELS     = ["llava:7b","minicpm-v"]
PROJECTS = {
    "BSH_Crack_Detection":  {"part":"BSH Metal Press Tool","hint":"Detect cracks on metal surface."},
    "Klinken_Rack_Position":{"part":"Press Clamp Rack","hint":"Detect multiple clamps extended."},
    "Klinken_Loaded_Rack":  {"part":"Press Clamp Rack Loaded","hint":"Detect too many clamps extended."},
    "Klinken_Empty_Rack":   {"part":"Press Clamp Rack Empty","hint":"Detect all clamps extended."},
}

async def run_model(model, img_b64, prompt, mime="image/jpeg"):
    if "qwen" in model.lower():
        payload={"model":model,"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":f"data:{mime};base64,{img_b64}"}},
            {"type":"text","text":prompt}]}],"stream":False,"temperature":0.0}
        endpoint=f"{OLLAMA_URL}/v1/chat/completions"
    else:
        payload={"model":model,"prompt":prompt,"images":[img_b64],"stream":False,"options":{"temperature":0.0}}
        endpoint=f"{OLLAMA_URL}/api/generate"
    try:
        t0=time.perf_counter()
        async with httpx.AsyncClient(timeout=180) as client:
            resp=await client.post(endpoint,json=payload)
            elapsed=round((time.perf_counter()-t0)*1000)
        raw=resp.json()["choices"][0]["message"]["content"].strip() if "qwen" in model.lower() else resp.json().get("response","").strip()
        m=re.search(r'\{.*?"verdict".*?\}',raw,re.DOTALL)
        if m:
            p=json.loads(m.group(0))
            return p.get("verdict","?").upper(), elapsed
        return ("ANOMALY" if "ANOMALY" in raw.upper() else "GOOD"), elapsed
    except Exception as e:
        return "ERROR", 0

GOOD_PROMPT = """You are a strict visual quality inspector for {part}.
Context: {hint}

Inspect this image carefully.

Respond ONLY with valid JSON:
{{"verdict": "ANOMALY" or "GOOD", "confidence": 0.0-1.0, "reason": "max 20 words"}}"""

async def test_good_images(model, proj_name, proj_cfg):
    good_dir = BASE_DIR/"data"/proj_name/"good"
    if not good_dir.exists():
        return 0, 0
    imgs = sorted(good_dir.glob("*.jpg"))
    if not imgs:
        return 0, 0
    correct, total = 0, 0
    prompt = GOOD_PROMPT.format(part=proj_cfg["part"], hint=proj_cfg["hint"])
    for img_path in imgs:
        with open(img_path,'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode()
        verdict, elapsed = await run_model(model, img_b64, prompt)
        ok = verdict == "GOOD"
        if ok: correct += 1
        total += 1
        status = "OK" if ok else "FP"
        print(f"    [{status}] {verdict} | {img_path.name[:35]} | {elapsed}ms")
    return correct, total

async def main():
    print("Testing all models on GOOD images (false positive check)")
    print("="*60)
    all_results = {}
    for model in MODELS:
        print(f"\nMODEL: {model}")
        print("="*60)
        model_results = {}
        for proj_name, proj_cfg in PROJECTS.items():
            good_dir = BASE_DIR/"data"/proj_name/"good"
            n_good = len(list(good_dir.glob("*.jpg"))) if good_dir.exists() else 0
            if n_good == 0:
                print(f"  SKIP {proj_name} — no good images")
                continue
            print(f"\n  [{proj_name}] ({n_good} good images)")
            correct, total = await test_good_images(model, proj_name, proj_cfg)
            acc = round(correct/total*100) if total else 0
            fp = total - correct
            print(f"  -> {correct}/{total} correctly GOOD ({acc}%) | FP: {fp}")
            model_results[proj_name] = {"correct":correct,"total":total,"accuracy":acc,"fp":fp}
        all_results[model] = model_results

    print(f"\n{'='*60}")
    print("SUMMARY — Good image accuracy (higher = fewer false positives)")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Project':<28} {'Accuracy':<10} {'FP'}")
    print("-"*60)
    for model, projs in all_results.items():
        for proj, r in projs.items():
            print(f"{model:<15} {proj:<28} {r['accuracy']}%{'':<5} {r['fp']}")

    out = BASE_DIR/"good_image_test_results.json"
    with open(out,'w') as f: json.dump(all_results, f, indent=2)
    print(f"\nSaved to good_image_test_results.json")

if __name__ == '__main__':
    asyncio.run(main())
