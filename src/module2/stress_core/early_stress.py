"""Раннее выявление скрытых очагов угнетения по каналу Red Edge.

Идея. При потере хлорофилла отражение в красном крае (Red Edge, B05) растёт, а NDRE
падает раньше, чем NDVI: на плотном пологе NDVI «насыщается» и ещё остаётся высоким.
Поэтому скрытым стрессом считаем пиксели, у которых NDRE заметно ниже нормы, а NDVI
пока в норме.

Нормой служит само поле на эту же дату (робастный z-score: медиана и MAD), что снимает
общий сезонный дрейф и влияние погоды. Если есть предыдущая дата, добавляются два правила динамики:

* ``temporal`` — резкое падение z-аномалии NDRE при почти неизменной аномалии NDVI (внезапный очаг);
* ``decline`` — NDRE в пикселе за интервал падает (в %) быстрее, чем в среднем по полю, и быстрее NDVI.
  Ловит постепенное угнетение участка, когда z-аномалии ещё малы. Пороги подобраны на сезоне 2026
  (8 дат, поле T37TEN): на 07 и 14 августа отмеченная площадь на 83–95 % попала в худшую четверть
  поля по обвалу NDVI к 24 августа (случайный выбор дал бы 25 %). Подробности — в README.

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
    # правило «опережающего спада» (если есть предыдущая дата, калибровка — сезон 2026)
    use_decline: bool = True
    decline_min_excess: float = 0.04  # NDRE пикселя упал (в долях) на >= 0.04 сильнее медианы поля
    decline_ratio: float = 1.5  # и сверх-падение NDRE >= 1.5 x сверх-падения NDVI
    min_area_px: int = 12
    edge_buffer_px: int = 1  # краевые смешанные пиксели не считаются очагом
    # флаг early_stress_alert (для дашборда M4)
    alert_area_fraction: float = 0.01  # скрытый стресс занимает >= 1% валидной площади поля
    alert_min_ndre_drop: float = 0.02  # или: средний NDRE упал на >= 0.02 относительно прошлой даты,
    alert_ratio: float = 1.5  # причём относительное падение NDRE >= 1.5 x падения NDVI
    alert_min_overlap: float = 0.5  # динамику сравниваем, если общих (без облаков на обеих датах) пикселей >= 50%


@dataclass
class EarlyStressResult:
    mask: np.ndarray  # bool: скрытый стресс
    z_ndre: np.ndarray  # робастные аномалии на эту дату (для следующей даты и отладки)
    z_ndvi: np.ndarray
    dz_ndre: np.ndarray | None = None  # изменение аномалий относительно прошлой даты
    dz_ndvi: np.ndarray | None = None
    by_rule: dict[str, np.ndarray] | None = None  # маски по правилам: spatial, temporal, decline


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
    prev_values: tuple[np.ndarray, np.ndarray] | None = None,
) -> EarlyStressResult:
    """Маска скрытого стресса.

    prev — пара (z_ndre, z_ndvi) предыдущей даты на той же сетке (из прошлого результата);
    prev_values — пара (NDRE, NDVI) предыдущей даты (значения индексов) для правила decline.
    """
    cfg = cfg or EarlyStressConfig()
    z_re, z_vi = anomalies(ndre, ndvi, cfg)
    rules: dict[str, np.ndarray] = {}

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
            rules["temporal"] = temporal
        rules["spatial"] = spatial
        if cfg.use_decline and prev_values is not None and prev_values[0].shape == z_re.shape:
            decline = relative_decline(ndre, ndvi, prev_values[0], prev_values[1], cfg)
            if decline is not None:
                mask = mask | decline
                rules["decline"] = decline

    finite = np.isfinite(z_re) & np.isfinite(z_vi)
    mask = remove_small_objects(np.where(finite, mask, False), cfg.min_area_px)
    by_rule = {k: v & mask for k, v in rules.items()}
    return EarlyStressResult(mask=mask, z_ndre=z_re, z_ndvi=z_vi, dz_ndre=d_re, dz_ndvi=d_vi, by_rule=by_rule)


def relative_decline(
    ndre: np.ndarray, ndvi: np.ndarray, prev_ndre: np.ndarray, prev_ndvi: np.ndarray, cfg: EarlyStressConfig
) -> np.ndarray | None:
    """Пиксели, где NDRE за интервал падает быстрее поля и быстрее NDVI (правило decline).

    Относительное изменение r = X_t / X_(t-1) - 1 считается по сглаженным значениям; из него вычитается
    медиана по полю (общий сезонный тренд, дымка, фенология всего поля). Условие: NDRE в пикселе упал,
    его сверх-падение не меньше decline_min_excess и в decline_ratio раз больше сверх-падения NDVI.
    None — если общих пикселей на двух датах меньше alert_min_overlap.
    """
    sm = lambda a: smooth_nan(np.asarray(a, dtype=np.float64), cfg.smooth_sigma_px)  # noqa: E731
    with np.errstate(invalid="ignore", divide="ignore"):
        r_re = sm(ndre) / sm(prev_ndre) - 1.0
        r_vi = sm(ndvi) / sm(prev_ndvi) - 1.0
    common = np.isfinite(r_re) & np.isfinite(r_vi) & np.isfinite(ndre) & np.isfinite(prev_ndre)
    n_cur = int((np.isfinite(ndre) & np.isfinite(ndvi)).sum())
    if not n_cur or common.sum() < cfg.alert_min_overlap * n_cur:
        return None
    e_re = r_re - np.median(r_re[common])
    e_vi = r_vi - np.median(r_vi[common])
    with np.errstate(invalid="ignore"):
        # r_re < 0: NDRE в пикселе действительно упал (в фазе роста «растёт медленнее поля» — не просадка)
        out = (r_re < 0) & (e_re <= -cfg.decline_min_excess) & (e_vi > e_re / cfg.decline_ratio)
    return out & common


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
    Пороги проверены на сезоне 2026 (см. README): (б) срабатывает 07 и 14 августа, до обвала NDVI 24 августа.
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
