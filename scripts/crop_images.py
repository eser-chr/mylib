import numpy as np
from pathlib import Path
from PIL import Image

IMAGE_DIR = Path("../Desktop/cuga-scans-paper")
OUTPUT_DIR = Path("../Desktop/cuga-scans-paper-cropped")
PADDING = 10

OUTPUT_DIR.mkdir(exist_ok=True)

image_paths = sorted(IMAGE_DIR.glob("*.png"))
if not image_paths:
    raise FileNotFoundError(f"No PNG files found in {IMAGE_DIR}")

for path in image_paths:
    img   = Image.open(path).convert("RGBA")
    alpha = np.array(img)[:, :, 3]

    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)

    top    = max(int(np.argmax(rows)) - PADDING, 0)
    bottom = min(int(len(rows) - np.argmax(rows[::-1])) + PADDING, img.height)
    left   = max(int(np.argmax(cols)) - PADDING, 0)
    right  = min(int(len(cols) - np.argmax(cols[::-1])) + PADDING, img.width)

    cropped = img.crop((left, top, right, bottom))
    cropped.save(OUTPUT_DIR / path.name)
    print(f"{path.name}: ({img.width}x{img.height}) -> ({cropped.width}x{cropped.height})")

print(f"\nDone. Saved to '{OUTPUT_DIR}/'")

