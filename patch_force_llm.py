"""Fix server.py and inspector.py for reliable model switching."""
import os

# Fix server.py
sp = r'C:\Users\gpngur01\Downloads\gdpr-ai2\server.py'
with open(sp, 'r', encoding='utf-8') as f:
    s = f.read()

old = """    # Override model if specified
    if model and model != 'dinov2':
        import inspector as insp
        insp.VISION_MODEL = model
        # Skip DINOv2, go straight to llava/qwen
        insp.FORCE_LLM = True
    else:
        import inspector as insp
        insp.FORCE_LLM = False"""

new = """    # Set model and FORCE_LLM flag before calling inspect
    import inspector as insp
    if model and model != 'dinov2':
        insp.VISION_MODEL = model
        insp.FORCE_LLM = True
    else:
        insp.FORCE_LLM = False"""

if old in s:
    s = s.replace(old, new)
    print("server.py patched OK")
else:
    print("server.py pattern not found — showing current content:")
    for i,l in enumerate(s.split('\n')[45:65], 46):
        print(f"{i}: {l}")

with open(sp, 'w', encoding='utf-8') as f:
    f.write(s)

# Fix inspector.py — ensure inspect_image reads FORCE_LLM at call time
ip = r'C:\Users\gpngur01\Downloads\gdpr-ai2\inspector.py'
with open(ip, 'r', encoding='utf-8') as f:
    c = f.read()

# Replace the if not FORCE_LLM check with a fresh module-level read
old2 = "    _force = force_llm or FORCE_LLM\n    if vision_model: VISION_MODEL_USE = vision_model\n    else: VISION_MODEL_USE = VISION_MODEL\n    if not _force:"
old3 = "    if not FORCE_LLM:"

# Try both patterns
if old2 in c:
    c = c.replace(old2, "    import inspector as _m\n    if not _m.FORCE_LLM:")
    print("inspector.py patched (pattern 2)")
elif old3 in c:
    c = c.replace(old3, "    import inspector as _m\n    if not _m.FORCE_LLM:")
    print("inspector.py patched (pattern 3)")
else:
    print("inspector.py — searching for pattern...")
    idx = c.find('async def inspect_image')
    print(repr(c[idx:idx+600]))

with open(ip, 'w', encoding='utf-8') as f:
    f.write(c)

import py_compile
for fp in [sp, ip]:
    try:
        py_compile.compile(fp, doraise=True)
        print(f"SYNTAX OK: {os.path.basename(fp)}")
    except Exception as e:
        print(f"ERROR in {os.path.basename(fp)}: {e}")
