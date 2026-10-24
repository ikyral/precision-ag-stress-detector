import json
import math
import os
from pathlib import Path

import numpy as np
import rasterio
from dotenv import load_dotenv
from pyproj import Transformer
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from shapely.geometry import Polygon, mapping
from urllib3 import Retry

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATE = "2026-08-07"
BANDS = ["B02_10m", "B03_10m", "B04_10m", "B05_20m", "B06_20m",
         "B07_20m", "B08_10m", "B8A_20m", "B11_20m", "B12_20m"]
STAC_URL = os.environ.get("CDSE_STAC_URL", "https://stac.dataspace.copernicus.eu/v1")
OUT_DIR = ROOT / "data" / "samples"

# --- контур поля ---
field = json.loads((OUT_DIR / "field_boundary.geojson").read_text(encoding="utf-8"))
ring = field["features"][0]["geometry"]["coordinates"][0]
lons = [p[0] for p in ring]
lats = [p[1] for p in ring]
bbox = [min(lons), min(lats), max(lons), max(lats)]

# --- поиск сцены на нужную дату ---
retry = Retry(total=6, backoff_factor=3, status_forcelist=[429, 502, 503, 504],
              allowed_methods=None)
cat = Client.open(STAC_URL, stac_io=StacApiIO(max_retries=retry))
items = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox,
                        datetime=f"{DATE}T00:00:00Z/{DATE}T23:59:59Z").items())
assert items, f"Нет сцен на {DATE}"
item = items[0]
print("Сцена:", item.id)

# Смещение +1000 есть у обработки с версии 04.00 (N0400 и новее)
baseline = int(item.id.split("_")[3][1:])
offset = 1000 if baseline >= 400 else 0
print("Смещение, которое вычитаем:", offset)


def vsi(asset):
    return item.assets[asset].href.replace("s3://", "/vsis3/")


os.environ["AWS_ACCESS_KEY_ID"] = os.environ["CDSE_S3_ACCESS_KEY"]
os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["CDSE_S3_SECRET_KEY"]
os.environ["AWS_S3_ENDPOINT"] = os.environ.get("CDSE_S3_ENDPOINT", "eodata.dataspace.copernicus.eu")
os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"
os.environ["AWS_HTTPS"] = "YES"

env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")

with env:
    # --- сетка 10 м: привязываем окно к пикселям эталонного канала B04 ---
    with rasterio.open(vsi("B04_10m")) as ref:
        crs, t0 = ref.crs, ref.transform
    print("CRS:", crs)

    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = to_utm.transform(lons, lats)
    poly = Polygon(zip(xs, ys))
    minx, miny, maxx, maxy = poly.bounds
    col0 = math.floor((minx - t0.c) / 10) - 1
    row0 = math.floor((t0.f - maxy) / 10) - 1
    col1 = math.ceil((maxx - t0.c) / 10) + 1
    row1 = math.ceil((t0.f - miny) / 10) + 1
    w, h = col1 - col0, row1 - row0
    tr = Affine(10, 0, t0.c + col0 * 10, 0, -10, t0.f - row0 * 10)
    print(f"Размер окна: {w} x {h} пикселей")

    def read(asset, method):
        with rasterio.open(vsi(asset)) as src, WarpedVRT(
                src, crs=crs, transform=tr, width=w, height=h,
                resampling=method) as vrt:
            return vrt.read(1)

    layers = []
    for b in BANDS:
        print("Читаю", b, flush=True)
        layers.append(read(b, Resampling.bilinear))
    scl = read("SCL_20m", Resampling.nearest)

# --- обрезка по контуру, отражение x10000 без смещения ---
inside = ~geometry_mask([mapping(poly)], out_shape=(h, w), transform=tr)
stack = np.stack(layers).astype("float32")
refl = np.clip(stack - offset, 1, 65535)       # минимум 1, чтобы не путать с nodata=0
scene = np.where(inside[None], refl, 0).astype("uint16")

bad = np.isin(scl, [3, 8, 9, 10]) & inside
print(f"SCL: облака/тени внутри поля: {bad.sum() / inside.sum() * 100:.1f}%")

prof = dict(driver="GTiff", count=len(BANDS), dtype="uint16", crs=crs,
            transform=tr, width=w, height=h, nodata=0, compress="deflate")
with rasterio.open(OUT_DIR / "sample_scene.tif", "w", **prof) as dst:
    dst.write(scene)
    dst.descriptions = tuple(b.split("_")[0] for b in BANDS)
    dst.update_tags(date=DATE, source=item.id, scale="reflectance*10000",
                    boa_offset_removed=str(offset))

# --- тестовая маска зон по терцилям NDVI ---
red, nir = scene[2].astype("float32"), scene[6].astype("float32")
ndvi = (nir - red) / (nir + red + 1e-6)
q1, q2 = np.percentile(ndvi[inside], [33.3, 66.6])
zones = np.zeros((h, w), "uint8")
zones[inside & (ndvi < q1)] = 1
zones[inside & (ndvi >= q1) & (ndvi < q2)] = 2
zones[inside & (ndvi >= q2)] = 3

zprof = {**prof, "count": 1, "dtype": "uint8", "nodata": None}
with rasterio.open(OUT_DIR / "sample_zones.tif", "w", **zprof) as dst:
    dst.write(zones, 1)

for name in ("sample_scene.tif", "sample_zones.tif"):
    size = (OUT_DIR / name).stat().st_size / 1e6
    print(f"{name}: {size:.1f} МБ")
print(f"Средний NDVI внутри поля: {ndvi[inside].mean():.2f}")