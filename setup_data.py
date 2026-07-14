"""Convert klinken TIF images from zip to JPG and copy to gdpr-ai2 data folder."""
import zipfile, os, io
from PIL import Image

LABELS = {
    'Klappenstellung 1': 'bad',
    'Klappenstellung 2': 'bad',
    'Klappenstellung 3': 'good',
    'Klappenstellung 4': 'good',
    'Klappenstellung 5': 'bad',
    'Klappenstellung 6': 'bad',
}

zip_path = r'C:\Users\gpngur01\Downloads\PressVisionLoop-main.zip'
out_good = r'C:\Users\gpngur01\Downloads\gdpr-ai2\data\klinken\good'
out_bad  = r'C:\Users\gpngur01\Downloads\gdpr-ai2\data\klinken\bad'

counts = {'good': 0, 'bad': 0}
with zipfile.ZipFile(zip_path) as z:
    for entry in z.namelist():
        if 'klinken-rack' in entry and entry.endswith('.tif'):
            label = None
            for pos, lbl in LABELS.items():
                if pos in entry:
                    label = lbl
                    break
            if not label:
                continue
            fname = os.path.basename(entry)[:12] + '.jpg'
            out_dir = out_good if label == 'good' else out_bad
            out_path = os.path.join(out_dir, fname)
            with z.open(entry) as f:
                img = Image.open(io.BytesIO(f.read()))
                img.convert('RGB').save(out_path, quality=85)
            counts[label] += 1

print(f"Klinken: {counts['good']} good, {counts['bad']} bad")
bsh_good = r'C:\Users\gpngur01\Downloads\gdpr-ai2\data\bsh\good'
bsh_bad  = r'C:\Users\gpngur01\Downloads\gdpr-ai2\data\bsh\bad'
print(f"BSH good: {len(os.listdir(bsh_good))}")
print(f"BSH bad:  {len(os.listdir(bsh_bad))}")
