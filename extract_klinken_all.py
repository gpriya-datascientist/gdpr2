"""
Extract ALL klinken images from zip including subfolders (115E/50E exposure variants).
This gives more diverse good images for better DINOv2 training.
"""
import zipfile, io, os, sys
from pathlib import Path
from PIL import Image

BASE = Path(r'C:\Users\gpngur01\Downloads\gdpr-ai2')
ZIP  = Path(r'C:\Users\gpngur01\Downloads\PressVisionLoop-main.zip')
GOOD = BASE / 'data' / 'klinken' / 'good'
BAD  = BASE / 'data' / 'klinken' / 'bad'

# Per klinken.py GESTELL_LABELS:
# KS1=bad, KS2=bad, KS3=good, KS4=good, KS5=bad, KS6=bad
LABELS = {
    'Klappenstellung 1': 'bad',
    'Klappenstellung 2': 'bad',
    'Klappenstellung 3': 'good',
    'Klappenstellung 4': 'good',
    'Klappenstellung 5': 'bad',
    'Klappenstellung 6': 'bad',
}

# Clear existing klinken images and re-extract all
for f in GOOD.iterdir():
    if f.suffix.lower() in {'.jpg','.jpeg','.png'}: f.unlink()
for f in BAD.iterdir():
    if f.suffix.lower() in {'.jpg','.jpeg','.png'}: f.unlink()

counts = {'good': 0, 'bad': 0}
with zipfile.ZipFile(ZIP) as z:
    for entry in z.namelist():
        if 'klinken-rack' not in entry or not entry.endswith('.tif'):
            continue
        # Determine which Klappenstellung this belongs to
        label = None
        ks_key = None
        for ks, lbl in LABELS.items():
            if ks in entry:
                label = lbl
                ks_key = ks.replace(' ','_')
                break
        if not label:
            continue
        # Extract and convert to grayscale JPG
        sha = Path(entry).stem[:12]
        # Include subfolder in name (115E or 50E)
        sub = ''
        if '115 E' in entry: sub = '_115E'
        elif '50 E' in entry: sub = '_50E'
        out_name = f'{ks_key}{sub}_{sha}.jpg'
        out_dir = GOOD if label == 'good' else BAD
        out_path = out_dir / out_name
        with z.open(entry) as f:
            img = Image.open(io.BytesIO(f.read())).convert('L').convert('RGB')
            img.save(out_path, quality=85)
        counts[label] += 1

print(f'Extracted: {counts["good"]} good, {counts["bad"]} bad')
print(f'Good dir: {len(list(GOOD.iterdir()))} files')
print(f'Bad dir:  {len(list(BAD.iterdir()))} files')
