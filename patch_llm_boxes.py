"""Fix inspect_image to pass image_path to _try_local."""
p = r'C:\Users\gpngur01\Downloads\gdpr-ai2\inspector.py'
with open(p,'r',encoding='utf-8') as f: content = f.read()

old = '    r = await _try_local(image_bytes, proj, mime, image_path)'
new = '    r = await _try_local(image_bytes, proj, mime, image_path=image_path)'
if old in content:
    content = content.replace(old, new)
    print("Fixed _try_local call with image_path")
else:
    idx = content.find('_try_local(image_bytes')
    print("Current:", repr(content[idx:idx+80]))

with open(p,'w',encoding='utf-8') as f: f.write(content)
import py_compile
try: py_compile.compile(p, doraise=True); print("SYNTAX OK")
except Exception as e: print("ERROR:", e)
