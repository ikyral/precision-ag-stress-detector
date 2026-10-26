"""Раннее выявление скрытых очагов угнетения по каналу Red Edge.

Идея. При потере хлорофилла отражение в красном крае (Red Edge, B05) растёт, а NDRE
падает раньше, чем NDVI: на плотном пологе NDVI «насыщается» и ещё остаётся высоким.
Поэтому скрытым стрессом считаем пиксели, у которых NDRE заметно ниже нормы, а NDVI
пока в норме.

Нормой служит само поле на эту же дату (робастный z-score: медиана и MAD), что снимает
общий сезонный дрейф и влияние погоды. Дополнительно, если есть предыдущая дата,
проверяется динамика: падение аномалии NDRE между датами при неизменной аномалии NDVI.

Ограничение: оценка относительная. Если угнетено больше половины поля, медиана
«уезжает» в сторону стресса и метод занижает долю очагов.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .filters import remove_small_objects, robust_z, smooth_nan


@dataclass
class EarlyStressConfig:
    smooth_sigma_px: float = 1.0
    # нижние границы масштаба аномалий (в единицах индекса): на однородном поле они не дают
    # микроскопическим отклонениям NDVI/NDRE выглядеть «значимыми»
    ndre_min_scale: float = 0.02
    ndvi_min_scale: float = 0.03
    # пространственное условие (одна дата)
    z_ndre_thr: float = 1.5  # NDRE-аномалия <= -z_ndre_thr
    z_ndvi_max_drop: float = 1.0  # а NDVI-аномалия > -z_ndvi_max_drop (ещё «норма»)
    # временное условие (если есть предыдущая дата)
    use_temporal: bool = True
    dz_ndre_thr: float = 1.0  # падение NDRE-аномалии между датами >= dz_ndre_thr
    dz_ndvi_max_drop: float = 0.5  # при этом NDVI-аномалия падает меньше, чем на это
    temporal_z_ndre_floor: float = 0.5  # и NDRE-аномалия уже <= -floor
    min_area_px: int = 12
    edge_buffer_px: int = 1  # краевые смешанные пиксели не считаются очагом
    # флаг early_stress_alert (для дашборда M4)
    alert_area_fraction: float = 0.01  # скрытый стресс занимает >= 1% валидной площади поля
    alert_min_ndre_drop: float = 0.02  # или: средний NDRE упал на >= 0.02 относительно прошлой даты,
    alert_ratio: float = 1.5  # причём относительное падение NDRE >= 1.5 x падения NDVI


@dataclass
class EarlyStressResult:
    mask: np.ndarray  # bool: скрытый стресс
    z_ndre: np.ndarray  # робастные аномалии на эту дату (для следующей даты и отладки)
    z_ndvi: np.ndarray
    dz_ndre: np.ndarray | None = None  # изменение аномалий относительно прошлой даты
    dz_ndvi: np.ndarray | None = None


def anomalies(
    ndre: np.ndarray, ndvi: np.ndarray, cfg: EarlyStressConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Робастные z-аномалии NDRE и NDVI после сглаживания шума."""
    return (
        robust_z(smooth_nan(ndre, cfg.smooth_sigma_px), cfg.ndre_min_scale),
        robust_z(smooth_nan(ndvi, cfg.smooth_sigma_px), cfg.ndvi_min_scale),
    )


def detect_hidden_stress(
    ndre: np.ndarray,
    ndvi: np.ndarray,
    cfg: EarlyStressConfig | None = None,
    prev: tuple[np.ndarray, np.ndarray] | None = None,
) -> EarlyStressResult:
    """Маска скрытого стресса.

    prev — пара (z_ndre, z_ndvi) предыдущей даты на той же сетке (из прошлого результата).
    """
    cfg = cfg or EarlyStressConfig()
    z_re, z_vi = anomalies(ndre, ndvi, cfg)

    with np.errstate(invalid="ignore"):
        spatial = (z_re <= -cfg.z_ndre_thr) & (z_vi > -cfg.z_ndvi_max_drop)
        mask = spatial
        d_re = d_vi = None
        if cfg.use_temporal and prev is not None and prev[0].shape == z_re.shape:
            d_re = z_re - prev[0]
            d_vi = z_vi - prev[1]
            temporal = (
                (d_re <= -cfg.dz_ndre_thr)
                & (d_vi > -cfg.dz_ndvi_max_drop)
                & (z_re <= -cfg.temporal_z_ndre_floor)
            )
            mask = spatial | temporal

    mask = np.where(np.isfinite(z_re) & np.isfinite(z_vi), mask, False)
    mask = remove_small_objects(mask, cfg.min_area_px)
    return EarlyStressResult(mask=mask, z_ndre=z_re, z_ndvi=z_vi, dz_ndre=d_re, dz_ndvi=d_vi)


def evaluate_alert(
    hidden_fraction: float | None,
    mean_ndre: float | None,
    mean_ndvi: float | None,
    prev_means: tuple[float | None, float | None] | None,
    cfg: EarlyStressConfig | None = None,
) -> dict:
    """Флаг early_stress_alert и причины срабатывания.

    Срабатывает, если (а) скрытый стресс занимает не меньше alert_area_fraction поля, или
    (б) есть предыдущая дата и NDRE упал быстрее NDVI: абсолютное падение среднего NDRE не меньше
    alert_min_ndre_drop, а относительное — не меньше alert_ratio от падения NDVI.
    Пороги не калибровались на реальных рядах и подлежат настройке на КТ4.
    """
    cfg = cfg or EarlyStressConfig()
    reasons: list[str] = []
    if hidden_fraction is not None and hidden_fraction >= cfg.alert_area_fraction:
        reasons.append("hidden_area")

    ndre_drop = ndvi_drop = None
    if prev_means is not None and None not in (prev_means[0], prev_means[1], mean_ndre, mean_ndvi):
        prev_ndre, prev_ndvi = prev_means
        ndre_drop = prev_ndre - mean_ndre  # положительное = упал
        ndvi_drop = prev_ndvi - mean_ndvi
        rel_ndre = ndre_drop / prev_ndre if prev_ndre else 0.0
        rel_ndvi = ndvi_drop / prev_ndvi if prev_ndvi else 0.0
        if ndre_drop >= cfg.alert_min_ndre_drop and rel_ndre >= cfg.alert_ratio * max(rel_ndvi, 0.005):
            reasons.append("ndre_faster_than_ndvi")
    return {
        "alert": bool(reasons),
        "reasons": reasons,
        "ndre_drop": None if ndre_drop is None else round(float(ndre_drop), 6),
        "ndvi_drop": None if ndvi_drop is None else round(float(ndvi_drop), 6),
    }
