import httpx, base64, json
import time

with open(r'C:\Users\gpngur01\Downloads\gdpr-ai2\data\klinken\bad\03a6617a2e08.jpg','rb') as f:
    img = f.read()
img_b64 = base64.b64encode(img).decode()

print(f"Image size: {len(img)/1024:.0f}KB")
print("Sending to Qwen2.5-VL (first call loads model, may take 2-3 min)...")
start = time.time()

payload = {
    'model': 'qwen2.5vl:7b',
    'messages': [{'role':'user','content':[
        {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+img_b64}},
        {'type':'text','text':'Reply ONLY with JSON: {"verdict":"GOOD" or "ANOMALY","reason":"brief description"}'}
    ]}],
    'stream':False,'temperature':0.0
}

r = httpx.post('http://localhost:11434/v1/chat/completions',
               json=payload, timeout=300)  # 5 minute timeout
elapsed = time.time()-start
print(f"Status: {r.status_code} | Time: {elapsed:.1f}s")
if r.status_code == 200:
    print('Response:', r.json()['choices'][0]['message']['content'][:400])
else:
    print('Error:', r.text[:300])
