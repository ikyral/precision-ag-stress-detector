"""PNG-превью для глазной проверки облаков: RGB | NDVI | маска."""
from __future__ import annotations

from pathlib import Path

import numpy as np

SCALE = 4                      # 1 пиксель сцены -> 4×4 пикселя превью
RGB_MAX = 0.18                 # единая растяжка для всех дат, отражение
OUTSIDE = (255, 255, 255)
NODATA = (255, 0, 255)         # NaN внутри поля
MASK_COLORS = {"scl": (255, 220, 0), "s2c": (255, 0, 255), "dilated": (0, 200, 255), "invalid": (90, 90, 90),
               "manual": (255, 120, 0)}


def _rgb(r, g, b) -> np.ndarray:
    st = np.stack([r, g, b], -1) / RGB_MAX
    return (np.clip(np.nan_to_num(st), 0, 1) ** (1 / 1.6) * 255).astype(np.uint8)


def _ndvi_colors(ndvi: np.ndarray) -> np.ndarray:
    # от коричневого (0.0) через жёлтый (0.4) к тёмно-зелёному (0.9)
    stops = np.array([0.0, 0.4, 0.9])
    cols = np.array([[140, 80, 30], [240, 220, 80], [0, 110, 40]], float)
    v = np.clip(np.nan_to_num(ndvi), stops[0], stops[-1])
    out = np.stack([np.interp(v, stops, cols[:, i]) for i in range(3)], -1)
    return out.astype(np.uint8)


def render(refl: np.ndarray, inside: np.ndarray, masks: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """refl [10, H, W] (порядок BANDS, можно с NaN) -> RGB-картинка из трёх панелей."""
    b, g, r, nir = refl[0], refl[1], refl[2], refl[6]
    nanm = ~np.isfinite(r) & inside
    rgb = _rgb(r, g, b)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - r) / (nir + r)
    nd = _ndvi_colors(ndvi)
    mk = rgb.copy() // 2 + 64
    for name in ("dilated", "scl", "s2c", "manual", "invalid"):
        if masks and name in masks:
            mk[masks[name] & inside] = MASK_COLORS[name]
    panels = []
    for p in (rgb, nd, mk):
        p = p.copy()
        p[nanm] = NODATA
        p[~inside] = OUTSIDE
        panels.append(np.repeat(np.repeat(p, SCALE, 0), SCALE, 1))
    gap = np.full((panels[0].shape[0], 6, 3), 255, np.uint8)
    return np.concatenate([panels[0], gap, panels[1], gap, panels[2]], 1)


def save_png(path: str | Path, img: np.ndarray, title: str) -> None:
    from PIL import Image, ImageDraw

    head = np.full((18, img.shape[1], 3), 255, np.uint8)
    im = Image.fromarray(np.concatenate([head, img], 0))
    ImageDraw.Draw(im).text((4, 3), title, fill=(0, 0, 0))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def contact_sheet(paths: list[Path], out: Path) -> None:
    """Все превью одной картинкой, по одному под другим."""
    from PIL import Image

    ims = [Image.open(p).convert("RGB") for p in paths]
    if not ims:
        return
    w, h = max(i.width for i in ims), sum(i.height for i in ims)
    sheet = Image.new("RGB", (w, h), "white")
    y = 0
    for im in ims:
        sheet.paste(im, (0, y))
        y += im.height
    sheet.save(out)
