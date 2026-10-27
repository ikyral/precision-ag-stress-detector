"""Чтение каналов из CDSE S3 (/vsis3/) окном прямо на целевую сетку."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from .grid import Grid

ROOT = Path(__file__).resolve().parents[3]  # корень репозитория

# GDAL сам повторяет HTTP-запросы при 429/5xx; поверх — повтор всего чтения с паузой
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="5",
                GDAL_HTTP_RETRY_DELAY="3", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".jp2")
RETRIES = 4
PAUSE_S = 10


def configure_s3() -> None:
    """Ключи CDSE S3 из .env -> переменные GDAL. Значения ключей не печатаются."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    missing = [k for k in ("CDSE_S3_ACCESS_KEY", "CDSE_S3_SECRET_KEY") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"В .env нет ключей: {', '.join(missing)} (см. .env.example)")
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ["CDSE_S3_ACCESS_KEY"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["CDSE_S3_SECRET_KEY"]
    os.environ["AWS_S3_ENDPOINT"] = os.environ.get("CDSE_S3_ENDPOINT", "eodata.dataspace.copernicus.eu")
    os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"
    os.environ["AWS_HTTPS"] = "YES"


def vsi(href: str) -> str:
    return href.replace("s3://", "/vsis3/", 1)


def _retry(fn, what: str):
    import rasterio

    for attempt in range(1, RETRIES + 1):
        try:
            with rasterio.Env(**GDAL_ENV):
                return fn()
        except rasterio.errors.RasterioError as exc:
            if attempt == RETRIES:
                raise
            pause = PAUSE_S * attempt
            print(f"    сбой чтения {what} ({exc}); повтор через {pause} с", flush=True)
            time.sleep(pause)


def tile_transform(href: str):
    """CRS и transform плитки (только заголовок файла)."""
    import rasterio

    def go():
        with rasterio.open(vsi(href)) as src:
            return src.crs.to_string(), src.transform
    return _retry(go, Path(href).name)


def read_on_grid(href: str, grid: Grid, resampling: str = "bilinear") -> np.ndarray:
    """Один канал, перепроецированный (bilinear / nearest) на сетку grid; читаются только нужные блоки."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    method = Resampling[resampling]

    def go():
        with rasterio.open(vsi(href)) as src, WarpedVRT(
                src, crs=grid.crs, transform=grid.transform, width=grid.width,
                height=grid.height, resampling=method) as vrt:
            return vrt.read(1)
    return _retry(go, Path(href).name)
