"""Restore Klinken_Loaded_Rack/bad from zip."""
import zipfile, io, os
from PIL import Image
from pathlib import Path

ZIP = r'C:\Users\gpngur01\Downloads\PressVisionLoop-main.zip'
DST = Path(r'C:\Users\gpngur01\Downloads\gdpr-ai2\data\Klinken_Loaded_Rack\bad')
DST.mkdir(parents=True, exist_ok=True)

# KS1, KS2, KS5 = bad for loaded rack
BAD_KS = ['Klappenstellung 1', 'Klappenstellung 2', 'Klappenstellung 5']

count = 0
with zipfile.ZipFile(ZIP) as z:
    for entry in z.namelist():
        if 'klinken-rack' not in entry or not entry.endswith('.tif'):
            continue
        ks_match = next((ks for ks in BAD_KS if ks in entry), None)
        if not ks_match:
            continue
        ks_num = ks_match.split()[-1]
        sha = Path(entry).stem[:12]
        # Include subfolder (115E/50E) in name
        sub = '_115E' if '115 E' in entry else ('_50E' if '50 E' in entry else '')
        out_name = f'Klappenstellung_{ks_num}{sub}_{sha}.jpg'
        out_path = DST / out_name
        with z.open(entry) as f:
            img = Image.open(io.BytesIO(f.read())).convert('L').convert('RGB')
            img.save(out_path, quality=85)
        count += 1

print(f"Restored: {count} images to Klinken_Loaded_Rack/bad/")
print(f"Files: {len(list(DST.glob('*.jpg')))}")
