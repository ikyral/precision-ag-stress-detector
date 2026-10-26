"""Общие вспомогательные функции: сглаживание с учётом NaN и робастные оценки."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def smooth_nan(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Гауссово сглаживание, в котором NaN не «просачиваются» в соседей.

    Нормированная свёртка: G(x * m) / G(m), где m — маска валидных пикселей.
    Невалидные пиксели остаются NaN.
    """
    arr = np.asarray(arr, dtype=np.float64)
    valid = np.isfinite(arr)
    if sigma <= 0:
        return np.where(valid, arr, np.nan)
    num = ndi.gaussian_filter(np.where(valid, arr, 0.0), sigma, mode="nearest")
    den = ndi.gaussian_filter(valid.astype(np.float64), sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~valid] = np.nan
    return out


def robust_center_scale(arr: np.ndarray) -> tuple[float, float]:
    """Медиана и робастный масштаб (1.4826 * MAD) по конечным значениям.

    Если MAD вырождается в ноль, берётся стандартное отклонение; если и оно ноль — 1.0.
    """
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(vals))
    scale = float(1.4826 * np.median(np.abs(vals - med)))
    if scale < 1e-9:
        scale = float(np.std(vals))
    if scale < 1e-9:
        scale = 1.0
    return med, scale


def robust_z(arr: np.ndarray, min_scale: float = 0.0) -> np.ndarray:
    """Робастный z-score относительно медианы поля на эту дату.

    min_scale — нижняя граница масштаба в единицах индекса. На очень однородном поле MAD
    почти ноль, и любое микроскопическое отклонение превращается в огромный z; порог
    не даёт шуму выглядеть значимым.
    """
    med, scale = robust_center_scale(arr)
    return (np.asarray(arr, dtype=np.float64) - med) / max(scale, min_scale)


def zscore(arr: np.ndarray) -> np.ndarray:
    """Обычная стандартизация (mean/std) по конечным значениям — для кластеризации."""
    a = np.asarray(arr, dtype=np.float64)
    vals = a[np.isfinite(a)]
    if vals.size == 0:
        return a
    std = float(vals.std())
    return (a - float(vals.mean())) / (std if std > 1e-9 else 1.0)


def remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Убирает 4-связные компоненты булевой маски площадью меньше min_size пикселей."""
    if min_size <= 1 or not mask.any():
        return mask.copy()
    labels, n = ndi.label(mask)
    sizes = ndi.sum(mask, labels, index=np.arange(1, n + 1))
    keep = np.concatenate([[False], sizes >= min_size])
    return keep[labels]


def core_mask(valid: np.ndarray, buffer_px: int) -> np.ndarray:
    """Валидные пиксели без краевого пояса шириной buffer_px (8-связность).

    Пиксели на границе поля или у дыры от облака смешанные: часть площади пикселя занимает соседний
    объект (дорога, другое поле, тень), поэтому их спектр искажён. Край самого растра краем поля не считается.
    """
    valid = np.asarray(valid, dtype=bool)
    if buffer_px <= 0:
        return valid.copy()
    return ndi.binary_erosion(
        valid, structure=ndi.generate_binary_structure(2, 2), iterations=int(buffer_px), border_value=1
    )
