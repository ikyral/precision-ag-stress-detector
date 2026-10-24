import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from urllib3 import Retry

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# Проверка, что ключи подхватились (сами значения не печатаем)
for k in ("CDSE_S3_ACCESS_KEY", "CDSE_S3_SECRET_KEY"):
    print(k, "OK" if os.environ.get(k) else "ОТСУТСТВУЕТ")

STAC_URL = os.environ.get("CDSE_STAC_URL", "https://stac.dataspace.copernicus.eu/v1")
DATES = "2026-06-01/2026-09-30"
MAX_CLOUD = 30  # предварительный фильтр по всей сцене, % (позже фильтруем по SCL внутри поля)

field = json.loads((ROOT / "data/samples/field_boundary.geojson").read_text(encoding="utf-8"))
geom = field["features"][0]["geometry"]

# Повторять запросы с паузой при 429 и 5xx
retry = Retry(total=6, backoff_factor=3,
              status_forcelist=[429, 502, 503, 504],
              allowed_methods=None)
cat = Client.open(STAC_URL, stac_io=StacApiIO(max_retries=retry))

# Лёгкий запрос: рамка вместо контура из 257 точек
lons = [p[0] for p in geom["coordinates"][0]]
lats = [p[1] for p in geom["coordinates"][0]]
bbox = [min(lons), min(lats), max(lons), max(lats)]

search = cat.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime=DATES,
    limit=100,
)
items = list(search.items())
print(f"\nНайдено сцен за {DATES}: {len(items)}")

rows = []
for it in items:
    cc = it.properties.get("eo:cloud_cover")
    rows.append((it.datetime.date().isoformat(), cc, it.id))
rows.sort()

print("\nДата        Облачность  ID")
for d, cc, i in rows:
    mark = "  <-- " if cc is not None and cc < MAX_CLOUD else ""
    print(f"{d}  {cc if cc is None else round(cc, 1):>8}   {i}{mark}")

clean = [r for r in rows if r[1] is not None and r[1] < MAX_CLOUD]
print(f"\nСцен с облачностью < {MAX_CLOUD}%: {len(clean)}")

# Для первой подходящей сцены покажем ассеты
if clean:
    first = next(it for it in items if it.id == clean[0][2])
    print(f"\nАссеты сцены {first.id}:")
    for name, a in first.assets.items():
        print(f"  {name:12s} {a.href}")