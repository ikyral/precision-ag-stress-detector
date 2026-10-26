"""Минимальный ввод/вывод GeoTIFF поверх tifffile (без GDAL/rasterio).

Читает и пишет многоканальные растры, сохраняя геопривязку (ModelPixelScale,
ModelTiepoint, GeoKeyDirectory) и nodata, чтобы маски зон ложились на исходную
сетку без перепроекции.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

TAG_PIXEL_SCALE = 33550
TAG_TIEPOINT = 33922
TAG_TRANSFORM = 34264
TAG_GEOKEYS = 34735
TAG_GEODOUBLE = 34736
TAG_GEOASCII = 34737
TAG_GDAL_NODATA = 42113

# GeoKey IDs
_KEY_MODEL_TYPE = 1024
_KEY_RASTER_TYPE = 1025
_KEY_GEOGRAPHIC_TYPE = 2048
_KEY_PROJECTED_TYPE = 3072
_KEY_LINEAR_UNITS = 3076

_MODEL_PROJECTED = 1
_MODEL_GEOGRAPHIC = 2
_UNIT_METRE = 9001


@dataclass
class GeoMeta:
    """Геопривязка растра в терминах GeoTIFF-тегов."""

    pixel_scale: tuple | None = None  # (sx, sy, sz)
    tiepoint: tuple | None = None  # (i, j, k, x, y, z)
    model_transformation: tuple | None = None
    geokeys: tuple | None = None
    geodoubles: tuple | None = None
    geoascii: str | None = None
    nodata: float | None = None

    # -- конструкторы -------------------------------------------------
    @classmethod
    def from_origin(
        cls, x_left: float, y_top: float, res_x: float, res_y: float, epsg: int,
        projected: bool = True,
    ) -> "GeoMeta":
        """Север сверху: верхний левый угол + размер пикселя + EPSG-код."""
        if projected:
            keys = [(_KEY_MODEL_TYPE, _MODEL_PROJECTED), (_KEY_RASTER_TYPE, 1),
                    (_KEY_PROJECTED_TYPE, epsg), (_KEY_LINEAR_UNITS, _UNIT_METRE)]
        else:
            keys = [(_KEY_MODEL_TYPE, _MODEL_GEOGRAPHIC), (_KEY_RASTER_TYPE, 1),
                    (_KEY_GEOGRAPHIC_TYPE, epsg)]
        flat = [1, 1, 0, len(keys)]
        for key_id, value in keys:
            flat += [key_id, 0, 1, value]
        return cls(
            pixel_scale=(float(res_x), float(res_y), 0.0),
            tiepoint=(0.0, 0.0, 0.0, float(x_left), float(y_top), 0.0),
            geokeys=tuple(flat),
        )

    # -- производные свойства ----------------------------------------
    def _keys(self) -> dict[int, int]:
        if not self.geokeys or len(self.geokeys) < 4:
            return {}
        n = int(self.geokeys[3])
        out = {}
        for i in range(n):
            key_id, loc, _count, value = self.geokeys[4 + 4 * i: 8 + 4 * i]
            if loc == 0:  # значение хранится прямо в записи
                out[int(key_id)] = int(value)
        return out

    @property
    def is_projected(self) -> bool:
        return self._keys().get(_KEY_MODEL_TYPE) == _MODEL_PROJECTED

    @property
    def epsg(self) -> int | None:
        keys = self._keys()
        code = keys.get(_KEY_PROJECTED_TYPE) or keys.get(_KEY_GEOGRAPHIC_TYPE)
        return int(code) if code and code != 32767 else None

    @property
    def pixel_area_m2(self) -> float | None:
        """Площадь пикселя в м². Известна только для проекции в метрах."""
        if not (self.pixel_scale and self.is_projected):
            return None
        unit = self._keys().get(_KEY_LINEAR_UNITS, _UNIT_METRE)
        if unit != _UNIT_METRE:
            return None
        return float(abs(self.pixel_scale[0] * self.pixel_scale[1]))


def read_geotiff(path: str | Path) -> tuple[np.ndarray, GeoMeta]:
    """Читает GeoTIFF -> (массив [bands, H, W], GeoMeta)."""
    with tifffile.TiffFile(str(path)) as tf:
        page = tf.pages[0]
        arr = tf.asarray()
        tags = page.tags

        def val(code):
            tag = tags.get(code)
            return None if tag is None else tag.value

        samples = int(page.samplesperpixel)
        if arr.ndim == 2:
            arr = arr[None]
        elif arr.ndim == 3 and int(page.planarconfig) != 2 and arr.shape[-1] == samples:
            arr = np.moveaxis(arr, -1, 0)  # чересстрочный (chunky) -> [bands, H, W]

        nodata = val(TAG_GDAL_NODATA)
        try:
            nodata = float(str(nodata).strip("\x00 ")) if nodata is not None else None
        except ValueError:
            nodata = None

        def tup(code):
            v = val(code)
            return None if v is None else tuple(float(x) for x in np.atleast_1d(v))

        geokeys = val(TAG_GEOKEYS)
        meta = GeoMeta(
            pixel_scale=tup(TAG_PIXEL_SCALE),
            tiepoint=tup(TAG_TIEPOINT),
            model_transformation=tup(TAG_TRANSFORM),
            geokeys=None if geokeys is None else tuple(int(x) for x in np.atleast_1d(geokeys)),
            geodoubles=tup(TAG_GEODOUBLE),
            geoascii=val(TAG_GEOASCII),
            nodata=nodata,
        )
    return arr, meta


def write_geotiff(
    path: str | Path, data: np.ndarray, meta: GeoMeta | None = None,
    nodata: float | None = None, compress: bool = True,
) -> None:
    """Пишет массив [H, W] или [bands, H, W] как GeoTIFF."""
    data = np.asarray(data)
    if data.ndim not in (2, 3):
        raise ValueError("ожидался массив [H, W] или [bands, H, W]")
    meta = meta or GeoMeta()

    extratags = []
    if meta.pixel_scale is not None:
        extratags.append((TAG_PIXEL_SCALE, "d", 3, tuple(meta.pixel_scale), True))
    if meta.tiepoint is not None:
        extratags.append((TAG_TIEPOINT, "d", 6, tuple(meta.tiepoint), True))
    if meta.model_transformation is not None:
        extratags.append((TAG_TRANSFORM, "d", 16, tuple(meta.model_transformation), True))
    if meta.geokeys is not None:
        extratags.append((TAG_GEOKEYS, "H", len(meta.geokeys), tuple(meta.geokeys), True))
    if meta.geodoubles is not None:
        extratags.append((TAG_GEODOUBLE, "d", len(meta.geodoubles), tuple(meta.geodoubles), True))
    if meta.geoascii is not None:
        extratags.append((TAG_GEOASCII, "s", 0, meta.geoascii, True))
    nd = nodata if nodata is not None else meta.nodata
    if nd is not None:
        extratags.append((TAG_GDAL_NODATA, "s", 0, repr(float(nd)) if data.dtype.kind == "f" else str(int(nd)), True))

    kwargs = dict(photometric="minisblack", extratags=extratags, metadata=None)
    if data.ndim == 3:
        kwargs["planarconfig"] = "separate"
    if compress:
        kwargs["compression"] = "deflate"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), data, **kwargs)
