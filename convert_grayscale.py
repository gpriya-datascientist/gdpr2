"""Convert all training images to grayscale for consistent DINOv2 training."""
from PIL import Image
from pathlib import Path

DIRS = [
    'data/bsh/good', 'data/bsh/bad',
    'data/klinken/good', 'data/klinken/bad',
    'data/casting/good', 'data/casting/bad',
]

base = Path(r'C:\Users\gpngur01\Downloads\gdpr-ai2')
total = 0
for d in DIRS:
    folder = base / d
    for f in folder.iterdir():
        if f.suffix.lower() in {'.jpg','.jpeg','.png'}:
            img = Image.open(f).convert('L').convert('RGB')
            img.save(f)
            total += 1
print(f"Converted {total} images to grayscale")
