"""Patch inspector.py to add XML-guided prompt for llava."""
p = r'C:\Users\gpngur01\Downloads\gdpr-ai2\inspector.py'
with open(p, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix _try_local call
content = content.replace(
    'r = await _try_local(image_bytes, proj, mime)',
    'r = await _try_local(image_bytes, proj, mime, image_path)')

# 2. Fix _try_local signature and add XML-guided prompt
old_fn = '''async def _try_local(image_bytes, proj, mime):
    start = time.perf_counter()
    try:
        img_b64 = base64.b64encode(image_bytes).decode()
        prompt = PROMPT_TEMPLATE.format(
            part_name=proj["part_name"],
            description_hint=proj["description_hint"])'''

new_fn = '''async def _try_local(image_bytes, proj, mime, image_path=None):
    start = time.perf_counter()
    try:
        img_b64 = base64.b64encode(image_bytes).decode()
        from annotation_utils import load_annotations
        xml_boxes = load_annotations(image_path) if image_path else None
        if xml_boxes:
            box_desc = []
            for i, b in enumerate(xml_boxes):
                box_desc.append(
                    f"  Region {i+1} ({b.get('name','defect')}): "
                    f"{int(b['x_min']*100)}-{int(b['x_max']*100)}% horizontal, "
                    f"{int(b['y_min']*100)}-{int(b['y_max']*100)}% vertical")
            prompt = (
                f"You are a strict visual quality inspector for {proj['part_name']}.\\n"
                f"EXPERT-ANNOTATED DEFECT LOCATIONS:\\n" + "\\n".join(box_desc) +
                "\\n\\nLook at EXACTLY these regions. Confirm defects there.\\n"
                "Respond ONLY with valid JSON:\\n"
                '{{"verdict": "ANOMALY" or "GOOD", "confidence": 0.95, '
                '"reason": "max 30 words", "defect_type": "crack or wrong_position or unknown"}}')
        else:
            prompt = PROMPT_TEMPLATE.format(
                part_name=proj["part_name"],
                description_hint=proj["description_hint"])'''

if old_fn in content:
    content = content.replace(old_fn, new_fn)
    print("PATCHED OK")
else:
    print("NOT FOUND - checking existing content...")
    idx = content.find('async def _try_local')
    print(repr(content[idx:idx+300]))

with open(p, 'w', encoding='utf-8') as f:
    f.write(content)
