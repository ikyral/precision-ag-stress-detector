"""Маска облаков и доля облаков в поле: SCL ∪ s2cloudless, отсев сцены, расширение маски."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCL_CLOUD = (3, 8, 9, 10)   # тень облака, облака средней/высокой вероятности, перистые
SCL_INVALID = (0, 1)        # NO_DATA (сцена не покрывает поле), дефект/насыщение

# s2cloudless: каналы модели (L1C) и параметры для 10 м (рекомендация авторов: average_over=22)
S2C_BANDS = ["B01", "B02", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12"]
S2C_THRESHOLD = 0.4
S2C_AVERAGE_OVER = 22
S2C_BUFFER_PX = 50          # контекст вокруг поля для свёртки, 500 м
CLDPRB_THRESHOLD = 40       # запасной вариант: MSK_CLDPRB из L2A, %


def percent_in_field(mask: np.ndarray, inside: np.ndarray) -> float:
    n = int(inside.sum())
    return 100.0 * int((mask & inside).sum()) / n if n else 100.0


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Расширение маски на px пикселей (квадратное окно 3×3, px раз)."""
    if px <= 0:
        return mask.copy()
    from scipy.ndimage import binary_dilation

    return binary_dilation(mask, structure=np.ones((3, 3), bool), iterations=px)


@dataclass
class CloudVerdict:
    invalid_percent: float          # SCL 0/1 в поле
    scl_percent: float              # SCL 3,8,9,10 в поле
    union_percent: float | None     # SCL ∪ s2cloudless (без расширения); None — не считалось
    nan_percent: float | None       # итоговая доля NaN в поле после расширения
    accepted: bool
    reason: str
    nan_mask: np.ndarray | None = None
    manual_percent: float = 0.0     # ручная маска (manual_masks.geojson) в поле, без расширения

    def as_dict(self) -> dict:
        r = lambda v: None if v is None else round(v, 2)  # noqa: E731
        return {"invalid_percent": r(self.invalid_percent), "scl_cloud_percent": r(self.scl_percent),
                "union_cloud_percent": r(self.union_percent), "manual_mask_percent": r(self.manual_percent),
                "nan_percent": r(self.nan_percent),
                "accepted": self.accepted, "reason": self.reason}


def scl_masks(scl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(облака/тени, невалидные) по SCL."""
    return np.isin(scl, SCL_CLOUD), np.isin(scl, SCL_INVALID)


def precheck(scl: np.ndarray, inside: np.ndarray, max_cloud: float) -> CloudVerdict | None:
    """Отсев только по SCL, до чтения L1C. None — сцену надо проверять дальше."""
    cloud, invalid = scl_masks(scl)
    inv, sc = percent_in_field(invalid, inside), percent_in_field(cloud, inside)
    if inv > max_cloud:
        return CloudVerdict(inv, sc, None, None, False, f"нет покрытия: SCL 0/1 {inv:.1f}% поля")
    if sc > max_cloud:
        # объединение с s2cloudless заведомо не меньше SCL — L1C можно не читать
        return CloudVerdict(inv, sc, None, None, False, f"облака: SCL {sc:.1f}% > {max_cloud:g}%")
    return None


def evaluate(scl: np.ndarray, extra_cloud: np.ndarray | None, inside: np.ndarray,
             max_cloud: float = 10.0, dilate_px: int = 2,
             manual: np.ndarray | None = None) -> CloudVerdict:
    """Решение по сцене.

    extra_cloud — маска s2cloudless (или MSK_CLDPRB) на той же сетке; None, если её нет.
    manual — ручная маска облаков/теней этой даты (manual_masks.geojson); None, если её нет.
    Сцена отбрасывается, если SCL 0/1 > max_cloud % поля («нет покрытия») или
    SCL{3,8,9,10} ∪ extra_cloud ∪ manual (без расширения) > max_cloud %. В NaN идут облака,
    расширенные на dilate_px, и невалидные пиксели SCL 0/1 (без расширения), только внутри поля.
    """
    pre = precheck(scl, inside, max_cloud)
    if pre is not None:
        return pre
    cloud, invalid = scl_masks(scl)
    union = cloud | extra_cloud if extra_cloud is not None else cloud.copy()
    man = manual if manual is not None else np.zeros_like(cloud)
    union |= man
    inv, sc, un, mp = (percent_in_field(m, inside) for m in (invalid, cloud, union, man))
    nan_mask = (dilate(union, dilate_px) | invalid) & inside
    nan = percent_in_field(nan_mask, inside)
    if un > max_cloud:
        return CloudVerdict(inv, sc, un, nan, False,
                            f"облака: SCL∪s2cloudless{'∪manual' if mp else ''} {un:.1f}% > {max_cloud:g}% "
                            f"(SCL {sc:.1f}%)", manual_percent=mp)
    return CloudVerdict(inv, sc, un, nan, True, "чистая", nan_mask, mp)


def s2cloudless_mask(l1c_refl: np.ndarray, threshold: float = S2C_THRESHOLD,
                     average_over: int = S2C_AVERAGE_OVER) -> tuple[np.ndarray, np.ndarray]:
    """l1c_refl: [10, H, W] отражение L1C (0..1) в порядке S2C_BANDS -> (вероятность, маска)."""
    from s2cloudless import S2PixelCloudDetector

    det = S2PixelCloudDetector(threshold=threshold, average_over=average_over,
                               dilation_size=0, all_bands=False)
    data = np.moveaxis(l1c_refl, 0, -1)[None].astype(np.float32)
    prob = det.get_cloud_probability_maps(data)[0]
    mask = det.get_mask_from_prob(prob[None])[0].astype(bool)
    return prob, mask
