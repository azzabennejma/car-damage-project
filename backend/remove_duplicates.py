import os
import cv2
import hashlib
from pathlib import Path

# =========================
# PROJECT ROOT SAFE PATH
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

IMAGES_DIR = BASE_DIR / "data" / "outputs" / "images"
LABELS_DIR = BASE_DIR / "data" / "outputs" / "labels"

# =========================
# IMAGE HASH (SAFE)
# =========================
def hash_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    # IMPORTANT: DO NOT resize (prevents false duplicates)
    return hashlib.md5(img.tobytes()).hexdigest()


# =========================
# REMOVE DUPLICATES (PAIR-BASED)
# =========================
def remove_duplicates():
    seen_hashes = set()

    removed_images = 0
    kept = 0

    if not IMAGES_DIR.exists():
        print(f"❌ Images folder not found: {IMAGES_DIR}")
        return

    images = os.listdir(IMAGES_DIR)

    for file in images:
        img_path = IMAGES_DIR / file
        label_path = LABELS_DIR / (Path(file).stem + ".txt")

        # ✅ STEP 1: validate pair exists
        if not img_path.exists() or not label_path.exists():
            continue

        h = hash_image(img_path)
        if h is None:
            continue

        # ✅ STEP 2: duplicate check
        if h in seen_hashes:
            try:
                os.remove(img_path)
                os.remove(label_path)
                removed_images += 1
                print(f"🗑️ Duplicate removed: {file}")
            except Exception as e:
                print(f"⚠️ Error removing {file}: {e}")
        else:
            seen_hashes.add(h)
            kept += 1

    print("\n====== DONE ======")
    print(f"Kept: {kept}")
    print(f"Removed: {removed_images}")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    remove_duplicates()