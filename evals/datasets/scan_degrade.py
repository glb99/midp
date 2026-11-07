"""Make scan-like copies of the synthetic documents, with the same labels.

    uv run python -m evals.datasets.scan_degrade

Each source image gets a small rotation, blur, noise, lower resolution and
JPEG compression, as a flatbed scan or a phone photo might. The copies go to
data/synthetic_scanned/<folder>/ with the labels unchanged, so a clean and a
degraded run of the same document can be compared directly.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCES = ["synthetic_delivery_notes", "synthetic_delivery_notes_untitled", "synthetic_invoices"]
OUT_ROOT = REPO_ROOT / "data" / "synthetic_scanned"


def degrade(image: Image.Image, rng: random.Random) -> Image.Image:
    angle = rng.uniform(-2.0, 2.0)
    out = image.rotate(angle, expand=False, fillcolor="white")
    out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.6, 1.2)))
    scale = rng.uniform(0.6, 0.8)
    out = out.resize((int(out.width * scale), int(out.height * scale)), Image.BILINEAR).resize(image.size, Image.BILINEAR)
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, 10, (out.height, out.width, 1))
    arr = np.clip(np.asarray(out).astype(float) + noise, 0, 255).astype("uint8")
    return Image.fromarray(arr)


def main() -> None:
    import io

    for folder in SOURCES:
        src = REPO_ROOT / "data" / folder
        dst = OUT_ROOT / folder
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / "labels.json", dst / "labels.json")
        rng = random.Random(folder)
        for png in sorted(src.glob("*.png")):
            image = degrade(Image.open(png).convert("RGB"), rng)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=rng.randint(45, 70))
            Image.open(io.BytesIO(buffer.getvalue())).convert("RGB").save(dst / png.name)
        print(f"wrote {len(list(dst.glob('*.png')))} scanned images to {dst}")


if __name__ == "__main__":
    main()
