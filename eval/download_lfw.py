"""
eval/download_lfw.py
====================
Downloads and prepares the LFW (Labeled Faces in the Wild) dataset
in the folder structure expected by eval/face_baseline.py:

    data/lfw/<person_name>/
        <person_name>_0001.jpg
        <person_name>_0002.jpg
        ...

Only downloads identities with >= MIN_IMAGES photos (default 10)
to ensure meaningful gallery/probe splits.

Run:
    python eval/download_lfw.py
"""

from __future__ import annotations

import os
import shutil
import tarfile
import urllib.request
from typing import Optional

LFW_URL = "https://ndownloader.figshare.com/files/5976015"
LFW_FIGSHARE_URL = "https://ndownloader.figshare.com/files/5976015"

# Also try these mirrors:
MIRRORS = [
    "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz",
    "https://ndownloader.figshare.com/files/5976015",
    "https://datasets.figure-eight.com/figure_eight_datasets/open-images/lfw-funneled.tgz",
]

MIN_IMAGES = 10          # Minimum images per identity
MAX_IDENTITIES = 50      # Cap to keep download manageable
DATA_DIR = "data"
LFW_TGZ = os.path.join(DATA_DIR, "lfw-funneled.tgz")
LFW_RAW_DIR = os.path.join(DATA_DIR, "lfw_funneled_raw")
LFW_DATASET_DIR = os.path.join(DATA_DIR, "lfw")


def download_file(url: str, dest: str) -> bool:
    """Download a file showing progress. Returns True on success."""
    try:
        print(f"  Downloading: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1024 * 64
            while True:
                data = resp.read(chunk)
                if not data:
                    break
                f.write(data)
                downloaded += len(data)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Progress: {pct:.1f}%  ({downloaded//1024//1024} MB / {total//1024//1024} MB)", end="", flush=True)
            print()
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def prepare_lfw_dataset() -> Optional[str]:
    """
    Downloads LFW-funneled, extracts it, and reorganizes it into the
    standard gallery/probe folder structure for eval/face_baseline.py.
    Returns the dataset directory path on success, None on failure.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LFW_DATASET_DIR, exist_ok=True)

    # 1. Download LFW tarball
    if not os.path.exists(LFW_TGZ):
        print("[LFW Download] Attempting download from mirrors...")
        success = False
        for url in MIRRORS:
            success = download_file(url, LFW_TGZ)
            if success:
                break

        if not success:
            print("\n[LFW Download] All mirrors failed.")
            print("  Please manually download LFW-funneled from:")
            print("  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz")
            print(f"  and save it to: {LFW_TGZ}")
            return None
    else:
        print(f"[LFW Download] Archive already exists: {LFW_TGZ}")

    # 2. Extract tarball
    if not os.path.exists(LFW_RAW_DIR):
        print(f"[LFW Download] Extracting to {LFW_RAW_DIR}...")
        os.makedirs(LFW_RAW_DIR, exist_ok=True)
        with tarfile.open(LFW_TGZ, "r:gz") as tar:
            tar.extractall(LFW_RAW_DIR)
        print("[LFW Download] Extraction complete.")
    else:
        print(f"[LFW Download] Already extracted at {LFW_RAW_DIR}")

    # 3. Find extracted LFW root (may be nested)
    lfw_inner = None
    for candidate in [
        os.path.join(LFW_RAW_DIR, "lfw_funneled"),
        os.path.join(LFW_RAW_DIR, "lfw-funneled"),
        LFW_RAW_DIR,
    ]:
        if os.path.isdir(candidate) and any(
            os.path.isdir(os.path.join(candidate, d)) for d in os.listdir(candidate)
        ):
            lfw_inner = candidate
            break

    if lfw_inner is None:
        print("[LFW Download] Could not find extracted LFW identity folders.")
        return None

    # 4. Collect identities with >= MIN_IMAGES
    identities = sorted([
        d for d in os.listdir(lfw_inner)
        if os.path.isdir(os.path.join(lfw_inner, d))
    ])

    qualified = []
    for person in identities:
        person_dir = os.path.join(lfw_inner, person)
        images = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(images) >= MIN_IMAGES:
            qualified.append((person, images))

    print(f"[LFW Download] Found {len(qualified)} identities with >= {MIN_IMAGES} images")

    # 5. Copy (or link) to structured dataset directory
    selected = qualified[:MAX_IDENTITIES]
    print(f"[LFW Download] Using top {len(selected)} identities (cap={MAX_IDENTITIES})")

    for person, images in selected:
        dest_person_dir = os.path.join(LFW_DATASET_DIR, person)
        os.makedirs(dest_person_dir, exist_ok=True)
        for img_fname in images:
            src = os.path.join(lfw_inner, person, img_fname)
            dst = os.path.join(dest_person_dir, img_fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    print(f"\n[LFW Download] Dataset ready at: {LFW_DATASET_DIR}")
    print(f"  Identities: {len(selected)}")
    total_imgs = sum(len(imgs) for _, imgs in selected)
    print(f"  Total images: {total_imgs}")
    print(f"\n  Run evaluation with:")
    print(f"    python -m eval.face_baseline --dataset_dir {LFW_DATASET_DIR}")

    return LFW_DATASET_DIR


if __name__ == "__main__":
    prepare_lfw_dataset()
