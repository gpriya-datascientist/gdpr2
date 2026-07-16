"""Fix qwen2.5vl to use OpenAI-compatible chat completions endpoint."""
p = r'C:\Users\gpngur01\Downloads\gdpr-ai2\inspector.py'
with open(p,'r',encoding='utf-8') as f: content = f.read()

old = '''        payload = {"model": VISION_MODEL, "prompt": prompt,
                   "images": [img_b64], "stream": False,
                   "options": {"temperature": 0.0}}
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            resp.raise_for_status()
            raw = resp.json().get("response","").strip()'''

new = '''        # qwen2.5vl needs OpenAI-compatible endpoint
        if "qwen" in VISION_MODEL.lower():
            payload = {"model": VISION_MODEL, "messages": [{"role":"user","content":[
                {"type":"image_url","image_url":{"url":f"data:{mime};base64,{img_b64}"}},
                {"type":"text","text":prompt}]}],
                "stream": False, "temperature": 0.0}
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(f"{OLLAMA_URL}/v1/chat/completions", json=payload)
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"].strip()
        else:
            payload = {"model": VISION_MODEL, "prompt": prompt,
                       "images": [img_b64], "stream": False,
                       "options": {"temperature": 0.0}}
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
                resp.raise_for_status()
                raw = resp.json().get("response","").strip()'''

if old in content:
    content = content.replace(old, new)
    print("Patched qwen endpoint")
else:
    print("Pattern not found")

with open(p,'w',encoding='utf-8') as f: f.write(content)

import py_compile
try:
    py_compile.compile(p, doraise=True)
    print("SYNTAX OK")
except Exception as e:
    print("ERROR:", e)
