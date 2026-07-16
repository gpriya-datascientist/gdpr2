"""Test minicpm-v on GOOD images only."""
import asyncio, base64, json, re, time, os
from pathlib import Path
import httpx
from dotenv import load_dotenv
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL","http://localhost:11434")
BASE_DIR   = Path(r"C:\Users\gpngur01\Downloads\gdpr-ai2")
MODEL      = "minicpm-v"

PROJECTS = {
    "BSH_Crack_Detection":   "BSH Metal Press Tool — smooth curved metal, bright glare is NORMAL.",
    "Klinken_Rack_Position": "Press Clamp Rack — max 1 clamp extended per tower is NORMAL.",
    "Klinken_Loaded_Rack":   "Press Clamp Rack Loaded — max 1 clamp extended per tower is NORMAL.",
    "Klinken_Empty_Rack":    "Press Clamp Rack Empty — all clamps retracted is NORMAL.",
}

PROMPT = """You are a strict visual quality inspector.
This is a KNOWN GOOD part: {hint}
Inspect this image. It should look normal with no defects.
Respond ONLY with valid JSON:
{{"verdict": "GOOD" or "ANOMALY", "confidence": 0.0-1.0, "reason": "max 20 words"}}"""

async def run(img_b64, prompt):
    payload = {"model": MODEL, "prompt": prompt, "images": [img_b64],
               "stream": False, "options": {"temperature": 0.0}}
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
    elapsed = round((time.perf_counter()-t0)*1000)
    raw = resp.json().get("response","").strip()
    m = re.search(r'\{.*?"verdict".*?\}', raw, re.DOTALL)
    if m:
        p = json.loads(m.group(0))
        return p.get("verdict","?").upper(), elapsed
    return ("ANOMALY" if "ANOMALY" in raw.upper() else "GOOD"), elapsed

async def main():
    print(f"Testing {MODEL} on GOOD images\n" + "="*50)
    total_c, total_t = 0, 0
    summary = []
    for proj_name, hint in PROJECTS.items():
        good_dir = BASE_DIR/"data"/proj_name/"good"
        imgs = sorted(good_dir.glob("*.jpg")) if good_dir.exists() else []
        if not imgs:
            print(f"SKIP {proj_name} — no good images"); continue
        print(f"\n[{proj_name}] ({len(imgs)} images)")
        correct = 0
        prompt = PROMPT.format(hint=hint)
        for img_path in imgs:
            with open(img_path,'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode()
            verdict, elapsed = await run(img_b64, prompt)
            ok = verdict == "GOOD"
            if ok: correct += 1
            status = "OK" if ok else "FP"
            print(f"  [{status}] {verdict} | {img_path.name[:35]} | {elapsed}ms")
        acc = round(correct/len(imgs)*100)
        fp  = len(imgs) - correct
        print(f"  -> {correct}/{len(imgs)} ({acc}%) | FP={fp}")
        total_c += correct; total_t += len(imgs)
        summary.append((proj_name, acc, correct, len(imgs), fp))

    print(f"\n{'='*50}")
    print("SUMMARY — minicpm-v on GOOD images")
    print(f"{'='*50}")
    print(f"{'Project':<28} {'Accuracy':<10} {'Correct':<10} {'FP'}")
    print("-"*52)
    for proj, acc, c, t, fp in summary:
        mark = "OK" if acc==100 else "!!"
        print(f"[{mark}] {proj:<26} {acc}%{'':<6} {c}/{t}{'':<6} {fp}")
    print(f"\nOverall: {total_c}/{total_t} ({round(total_c/total_t*100)}%)")

asyncio.run(main())
