"""
Download Kaggle casting dataset and expand training data for better DINOv2 accuracy.
Dataset: ravirajsinh45/real-life-industrial-dataset-of-casting-product
good = ok_front, bad = def_front
"""
import os, sys, shutil, subprocess
from pathlib import Path

BASE_DIR = Path(r"C:\Users\gpngur01\Downloads\gdpr-ai2")
CACHE_DIR = BASE_DIR / "data" / "downloads" / "casting"
GOOD_DIR  = BASE_DIR / "data" / "bsh" / "good"
BAD_DIR   = BASE_DIR / "data" / "bsh" / "bad"

# Load .env
from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

KAGGLE_USER = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY  = os.getenv("KAGGLE_KEY", "")

def check_kaggle():
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        print(f"Found kaggle.json at {kaggle_json}")
        return True
    if KAGGLE_USER and KAGGLE_KEY:
        # Create kaggle.json from .env
        kaggle_json.parent.mkdir(exist_ok=True)
        import json
        kaggle_json.write_text(json.dumps({"username": KAGGLE_USER, "key": KAGGLE_KEY}))
        kaggle_json.chmod(0o600)
        print(f"Created kaggle.json from .env credentials")
        return True
    print("No Kaggle credentials found!")
    return False

def download_casting():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    marker = CACHE_DIR / ".extracted"
    if marker.exists():
        print("Already downloaded, skipping...")
        return CACHE_DIR
    print("Downloading casting dataset from Kaggle (~1.5GB)...")
    kaggle_exe = str(Path(sys.executable).parent / "kaggle.exe")
    if not Path(kaggle_exe).exists():
        kaggle_exe = shutil.which("kaggle") or "kaggle"
    subprocess.run([kaggle_exe, "datasets", "download",
        "-d", "ravirajsinh45/real-life-industrial-dataset-of-casting-product",
        "-p", str(CACHE_DIR), "--unzip"], check=True)
    marker.touch()
    print("Download complete!")
    return CACHE_DIR

def find_images(root, folder_name):
    imgs = []
    for p in Path(root).rglob("*"):
        if p.is_file() and p.parent.name == folder_name and p.suffix.lower() in {".jpg",".jpeg",".png"}:
            imgs.append(p)
    return sorted(imgs)

def copy_images(src_list, dst_dir, prefix, limit=20):
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for i, src in enumerate(src_list[:limit]):
        dst = dst_dir / f"{prefix}_{i:04d}{src.suffix}"
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    return copied

if __name__ == "__main__":
    print("=== PressVision2 Dataset Expander ===\n")
    if not check_kaggle():
        sys.exit(1)

    print(f"Current BSH good: {len(list(GOOD_DIR.glob('*.jpg')))}")
    print(f"Current BSH bad:  {len(list(BAD_DIR.glob('*.jpg')))}\n")

    cache = download_casting()

    ok_images  = find_images(cache, "ok_front")
    def_images = find_images(cache, "def_front")
    print(f"Found: {len(ok_images)} ok_front, {len(def_images)} def_front")

    # Copy top 40 of each to expand training set
    n_good = copy_images(ok_images,  GOOD_DIR, "casting_good", limit=40)
    n_bad  = copy_images(def_images, BAD_DIR,  "casting_bad",  limit=40)
    print(f"Added: {n_good} good + {n_bad} bad casting images")

    print(f"\nFinal BSH good: {len(list(GOOD_DIR.glob('*.*')))}")
    print(f"Final BSH bad:  {len(list(BAD_DIR.glob('*.*')))}")
    print("\nDone! Restart server to retrain DINOv2 with expanded dataset.")
