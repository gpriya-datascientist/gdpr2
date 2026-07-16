import asyncio,base64,json,re,httpx,os
from pathlib import Path
os.chdir(r'C:\Users\gpngur01\Downloads\gdpr-ai2')
BASE=Path(r'C:\Users\gpngur01\Downloads\gdpr-ai2')
URL='http://localhost:11434'
MODELS=['llava:7b','minicpm-v','qwen2.5vl:7b']
PROJECTS={'BSH_Crack_Detection':'BSH Metal Press Tool - smooth metal, bright glare NORMAL','Klinken_Rack_Position':'Press Clamp Rack - max 1 clamp extended is NORMAL','Klinken_Loaded_Rack':'Loaded Rack - max 1 clamp extended is NORMAL','Klinken_Empty_Rack':'Empty Rack - all clamps retracted is NORMAL'}
PROMPT='You are a quality inspector. This is a KNOWN GOOD part: {hint}\nRespond ONLY with JSON: {{"verdict": "GOOD" or "ANOMALY", "confidence": 0.95}}'
async def infer(model,img_b64,hint):
    p=PROMPT.format(hint=hint)
    if 'qwen' in model:
        pl={"model":model,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img_b64}"}},{"type":"text","text":p}]}],"stream":False,"temperature":0.0}
        ep=f"{URL}/v1/chat/completions"
    else:
        pl={"model":model,"prompt":p,"images":[img_b64],"stream":False,"options":{"temperature":0.0}}
        ep=f"{URL}/api/generate"
    async with httpx.AsyncClient(timeout=180) as c:
        r=await c.post(ep,json=pl)
    raw=r.json()["choices"][0]["message"]["content"] if "qwen" in model else r.json().get("response","")
    m=re.search(r'\{.*?"verdict".*?\}',raw,re.DOTALL)
    v=json.loads(m.group(0)).get("verdict","?").upper() if m else ("ANOMALY" if "ANOMALY" in raw.upper() else "GOOD")
    return v
async def main():
    print("Model,Project,Correct,Total,Accuracy,FP")
    for model in MODELS:
        for proj,hint in PROJECTS.items():
            gdir=BASE/'data'/proj/'good'
            imgs=list(gdir.glob('*.jpg')) if gdir.exists() else []
            if not imgs: continue
            correct=0
            for img in imgs:
                b64=base64.b64encode(open(img,'rb').read()).decode()
                v=await infer(model,b64,hint)
                if v=='GOOD': correct+=1
                else: print(f"  FP: {model} {proj} {img.name[:25]}")
            acc=round(correct/len(imgs)*100)
            print(f"{model},{proj},{correct},{len(imgs)},{acc}%,{len(imgs)-correct}")
asyncio.run(main())
