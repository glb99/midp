"""Generate sales invoices with known labels, for evaluating the invoice config.

    uv run python -m evals.datasets.synthetic_invoices --count 40

Each invoice is a PNG in data/synthetic_invoices/ with a slight blur and noise,
so it looks a little like a scan. Its labels go to labels.json, in the same
shape as the receipt config. Nothing here comes from a real company.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "synthetic_invoices"
FONT = Path("C:/Windows/Fonts/arial.ttf")

SUPPLIERS = ["Talleres Norte S.L.", "Distribuciones Sur S.A.", "Papelería Central", "Informática Levante", "Suministros Oeste"]
ITEMS = ["Tornillos M6 (caja)", "Papel A4 (paquete)", "Cable HDMI 2 m", "Servicio de mantenimiento", "Tinta negra 500 ml", "Monitor 24 pulgadas", "Teclado inalámbrico", "Licencia anual de software"]
TAX_RATE = 0.21


def _invoice(rng: random.Random, index: int) -> tuple[dict, dict]:
    issued = date(2025, 1, 1) + timedelta(days=rng.randint(0, 540))
    lines = []
    for description in rng.sample(ITEMS, k=rng.randint(1, 5)):
        quantity = rng.randint(1, 5)
        unit = rng.randint(5, 200)
        lines.append({"description": description, "quantity": quantity, "amount": quantity * unit})
    subtotal = sum(line["amount"] for line in lines)
    tax = round(subtotal * TAX_RATE)
    labels = {
        "number": f"F{issued.year}-{index:04d}",
        "date": issued.strftime("%d/%m/%Y"),
        "supplier": rng.choice(SUPPLIERS),
        "lines": lines,
        "subtotal": subtotal,
        "tax": tax,
        "total": subtotal + tax,
    }
    return labels, {}


def _render(labels: dict, rng: random.Random, title: str | None = "FACTURA", prices: bool = True) -> Image.Image:
    image = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(FONT), 34) if FONT.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(FONT), 28) if FONT.exists() else ImageFont.load_default()

    if title:
        draw.text((80, 80), title, fill="black", font=font)
    draw.text((80, 150), f"Número: {labels['number']}", fill="black", font=small)
    draw.text((80, 190), f"Fecha: {labels['date']}", fill="black", font=small)
    draw.text((80, 230), f"Proveedor: {labels['supplier']}", fill="black", font=small)

    y = 340
    draw.text((80, y), "Descripción", fill="black", font=small)
    draw.text((700, y), "Cant.", fill="black", font=small)
    draw.text((900, y), "Importe", fill="black", font=small)
    y += 60
    for line in labels["lines"]:
        draw.text((80, y), line["description"], fill="black", font=small)
        draw.text((700, y), str(line["quantity"]), fill="black", font=small)
        draw.text((900, y), f"{line['amount']} EUR", fill="black", font=small)
        y += 50

    y += 60
    for caption, value in ([("Base imponible", labels["subtotal"]), ("IVA 21%", labels["tax"]), ("Total", labels["total"])] if prices else []):
        draw.text((600, y), caption, fill="black", font=small)
        draw.text((900, y), f"{value} EUR", fill="black", font=small)
        y += 50

    image = image.filter(ImageFilter.GaussianBlur(radius=0.8))
    noise = [rng.randint(-12, 12) for _ in range(256)]
    return image.point(lambda v: max(0, min(255, v + noise[v % 256])))


def generate_delivery_notes(count: int, seed: int, titled: bool) -> Path:
    """Albaranes: the same layout and items as invoices, but no prices or tax.

    With titled=False the header says nothing about the type, so the model
    has to decide from the content alone.
    """
    rng = random.Random(seed)
    out_dir = REPO_ROOT / "data" / ("synthetic_delivery_notes" if titled else "synthetic_delivery_notes_untitled")
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_out = {}
    for index in range(count):
        labels, _ = _invoice(rng, index)
        for key in ("subtotal", "tax", "total"):
            labels[key] = None
        name = f"{index:04d}.png"
        _render(labels, rng, title="ALBARÁN DE ENTREGA" if titled else None, prices=False).save(out_dir / name)
        labels_out[name] = labels
    (out_dir / "labels.json").write_text(json.dumps(labels_out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {count} delivery notes to {out_dir}")
    return out_dir


def generate(count: int, seed: int = 0) -> None:
    rng = random.Random(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels_out = {}
    for index in range(count):
        labels, _ = _invoice(rng, index)
        name = f"{index:04d}.png"
        _render(labels, rng).save(OUT_DIR / name)
        labels_out[name] = labels
    (OUT_DIR / "labels.json").write_text(json.dumps(labels_out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {count} invoices to {OUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="evals.datasets.synthetic_invoices")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate(args.count, args.seed)


if __name__ == "__main__":
    main()
