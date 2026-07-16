import asyncio,base64,json,re,httpx,os,sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
BASE=Path(r'C:\Users\gpngur01\Downloads\gdpr-ai2')
URL='http://localhost:11434'
PROJECTS={'BSH_Crack_Detection':'BSH Metal Press Tool - smooth metal surface, bright glare is NORMAL, no defects expected','Klinken_Rack_Position':'Press Clamp Rack - max 1 clamp extended per tower is NORMAL','Klinken_Loaded_Rack':'Loaded Press Clamp Rack - max 1 clamp extended per tower is NORMAL','Klinken_Empty_Rack':'Empty Press Clamp Rack - all clamps retracted is NORMAL'}
PROMPT='You are a quality inspector. This is a KNOWN GOOD part: {hint}\nThis image should show NO defects. Inspect carefully.\nRespond ONLY with valid JSON: {{"verdict": "GOOD" or "ANOMALY", "confidence": 0.95}}'
async def infer(model,img_b64,hint):
    p=PROMPT.format(hint=hint)
    if 'qwen' in model:
        pl={"model":model,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img_b64}"}},{"type":"text","text":p}]}],"stream":False,"temperature":0.0}
        ep=f"{URL}/v1/chat/completions"
    else:
        pl={"model":model,"prompt":p,"images":[img_b64],"stream":False,"options":{"temperature":0.0}}
        ep=f"{URL}/api/generate"
    try:
        async with httpx.AsyncClient(timeout=180) as c:
            r=await c.post(ep,json=pl)
        raw=r.json()["choices"][0]["message"]["content"] if "qwen" in model else r.json().get("response","")
        m=re.search(r'\{.*?"verdict".*?\}',raw,re.DOTALL)
        v=json.loads(m.group(0)).get("verdict","?").upper() if m else ("ANOMALY" if "ANOMALY" in raw.upper() else "GOOD")
        return v
    except: return "ERROR"
async def main():
    models=['llava:7b','minicpm-v','qwen2.5vl:7b']
    results={}
    for model in models:
        results[model]={}
        print(f"\nMODEL: {model}")
        print("-"*50)
        for proj,hint in PROJECTS.items():
            gdir=BASE/'data'/proj/'good'
            imgs=list(gdir.glob('*.jpg')) if gdir.exists() else []
            if not imgs: continue
            correct=0
            for img in imgs:
                b64=base64.b64encode(open(img,'rb').read()).decode()
                v=await infer(model,b64,hint)
                ok=v=='GOOD'
                if ok: correct+=1
                print(f"  {'OK' if ok else 'FP'} | {v} | {img.name[:30]}")
            acc=round(correct/len(imgs)*100)
            fp=len(imgs)-correct
            results[model][proj]={'acc':acc,'correct':correct,'total':len(imgs),'fp':fp}
            print(f"  => {proj}: {correct}/{len(imgs)} ({acc}%) FP={fp}")
    print("\n"+"="*60)
    print("FINAL SUMMARY — Good image accuracy (false positive rate)")
    print("="*60)
    print(f"{'Model':<16}{'Project':<28}{'Acc':<8}{'Correct':<10}{'FP'}")
    print("-"*65)
    for model,projs in results.items():
        for proj,r in projs.items():
            mark="OK" if r['acc']==100 else "!!"
            print(f"[{mark}] {model:<14}{proj:<28}{r['acc']}%{'':<4}{r['correct']}/{r['total']}{'':<6}{r['fp']}")
    with open(BASE/'good_results.json','w') as f: json.dump(results,f,indent=2)
    print("\nSaved to good_results.json")
asyncio.run(main())
