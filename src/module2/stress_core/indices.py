"""Вегетационные индексы: NDVI, NDRE, MSAVI2, NDMI.

Формулы совпадают с каталогом Awesome Spectral Indices (ASI), имена каналов — как в ASI:
    R   — красный (Sentinel-2 B04)
    RE1 — Red Edge 1 (B05)
    N   — ближний ИК (B08)
    S1  — SWIR 1 (B11)

Основной бэкенд — NumPy (работает без внешних зависимостей). Бэкенд ``spyndex``
подключается опционально и нужен, в первую очередь, для перекрёстной проверки.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

INDEX_NAMES = ("NDVI", "NDRE", "MSAVI2", "NDMI")

# какие каналы нужны индексу
REQUIRED_BANDS: dict[str, tuple[str, ...]] = {
    "NDVI": ("N", "R"),
    "NDRE": ("N", "RE1"),
    "MSAVI2": ("N", "R"),
    "NDMI": ("N", "S1"),
}

# имя индекса в каталоге ASI / spyndex
# коды в каталоге ASI / spyndex; пробуем по порядку (в ASI NDRE называется NDREI)
ASI_NAME = {
    "NDVI": ("NDVI",),
    "NDRE": ("NDREI", "NDRE"),
    "MSAVI2": ("MSAVI", "MSAVI2"),
    "NDMI": ("NDMI",),
}


def _spyndex_compute(spyndex, name, params):
    """Вызывает spyndex.computeIndex, перебирая допустимые коды индекса из ASI_NAME."""
    last = None
    for code in ASI_NAME[name]:
        try:
            return spyndex.computeIndex(code, params=params)
        except Exception as exc:  # spyndex бросает общий Exception для неизвестного кода
            if "not a valid" not in str(exc):
                raise
            last = exc
    raise ValueError(f"{name}: ни один код из {ASI_NAME[name]} не найден в spyndex") from last

REFLECTANCE_MAX = 1.5  # выше — облака/блики, пиксель считаем невалидным


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), деление на ноль -> NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / (a + b)
    out[~np.isfinite(out)] = np.nan
    return out


def ndvi(N, R):
    return _nd(N, R)


def ndre(N, RE1):
    return _nd(N, RE1)


def ndmi(N, S1):
    return _nd(N, S1)


def msavi2(N, R):
    """MSAVI2 = 0.5 * (2N + 1 - sqrt((2N + 1)^2 - 8 (N - R)))."""
    with np.errstate(invalid="ignore"):
        radicand = (2.0 * N + 1.0) ** 2 - 8.0 * (N - R)
        out = 0.5 * (2.0 * N + 1.0 - np.sqrt(radicand))
    out[~np.isfinite(out)] = np.nan
    return out


_NUMPY_FUNCS = {
    "NDVI": lambda b: ndvi(b["N"], b["R"]),
    "NDRE": lambda b: ndre(b["N"], b["RE1"]),
    "MSAVI2": lambda b: msavi2(b["N"], b["R"]),
    "NDMI": lambda b: ndmi(b["N"], b["S1"]),
}


def valid_mask(bands: Mapping[str, np.ndarray], names: Sequence[str] = INDEX_NAMES) -> np.ndarray:
    """Пиксель валиден, если все нужные каналы конечны и 0 < reflectance <= 1.5."""
    needed = sorted({b for n in names for b in REQUIRED_BANDS[n]})
    mask = None
    for b in needed:
        arr = np.asarray(bands[b])
        m = np.isfinite(arr) & (arr > 0) & (arr <= REFLECTANCE_MAX)
        mask = m if mask is None else (mask & m)
    return mask


def compute_indices(
    bands: Mapping[str, np.ndarray],
    names: Sequence[str] = INDEX_NAMES,
    backend: str = "numpy",
) -> dict[str, np.ndarray]:
    """Считает индексы по каналам отражения (0..1).

    Невалидные пиксели (см. :func:`valid_mask`) получают NaN во всех индексах.
    Возвращает float32-массивы той же формы, что и входные каналы.
    """
    if backend not in ("numpy", "spyndex"):
        raise ValueError("backend должен быть 'numpy' или 'spyndex'")
    missing = {b for n in names for b in REQUIRED_BANDS[n]} - set(bands)
    if missing:
        raise KeyError(f"не хватает каналов: {sorted(missing)}")

    arrs = {k: np.asarray(v, dtype=np.float64) for k, v in bands.items()}
    valid = valid_mask(arrs, names)

    out: dict[str, np.ndarray] = {}
    if backend == "numpy":
        for n in names:
            out[n] = _NUMPY_FUNCS[n](arrs)
    else:
        import spyndex  # импорт здесь, чтобы пакет работал без spyndex

        for n in names:
            params = {b: arrs[b] for b in REQUIRED_BANDS[n]}
            out[n] = np.asarray(_spyndex_compute(spyndex, n, params), dtype=np.float64)

    for n in names:
        res = np.array(out[n], dtype=np.float64, copy=True)
        res[~valid] = np.nan
        out[n] = res.astype(np.float32)
    return out


def compare_with_spyndex(
    bands: Mapping[str, np.ndarray], names: Sequence[str] = INDEX_NAMES
) -> dict[str, float]:
    """Максимальное абсолютное расхождение NumPy-реализации и spyndex по индексам."""
    a = compute_indices(bands, names, backend="numpy")
    b = compute_indices(bands, names, backend="spyndex")
    return {n: float(np.nanmax(np.abs(a[n] - b[n]))) for n in names}
