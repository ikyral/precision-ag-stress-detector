"""Сегментация поля на зоны управления (Management Zones).

Конвейер:
  1. признаки (NDVI, NDRE, MSAVI2, NDMI) сглаживаются с учётом NaN (подавление шума);
  2. каждый признак стандартизируется (mean/std) и ограничивается по модулю;
  3. KMeans на подвыборке пикселей (k задан или выбирается по силуэту в диапазоне 3–5);
  4. кластеры упорядочиваются по «жизненной силе» (среднее z-признаков): 1 — острый стресс,
     N — норма;
  5. пространственная регуляризация: мажоритарный фильтр + слияние мелких пятен
     с самым частым соседом;
  6. нумерация делается подряд (1..N), 0 — нет данных.

Классы относительны: они описывают, какие участки хуже/лучше других на этом поле в эту дату,
а не абсолютные пороги.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .filters import smooth_nan, zscore

CLASS_LABELS: dict[int, list[str]] = {
    1: ["однородное поле"],
    2: ["стресс", "норма"],
    3: ["сильный стресс", "умеренный стресс", "норма"],  # контракт M2 -> M3
    4: ["острый стресс", "умеренный стресс", "слабый стресс", "норма"],
    5: ["острый стресс", "сильный стресс", "умеренный стресс", "слабый стресс", "норма"],
}


def class_labels(n: int) -> list[str]:
    if n in CLASS_LABELS:
        return CLASS_LABELS[n]
    return ["острый стресс"] + [f"класс {i}" for i in range(2, n)] + ["норма"]


@dataclass
class ZoningConfig:
    features: tuple[str, ...] = ("NDVI", "NDRE", "MSAVI2", "NDMI")
    n_classes: int | None = 3  # 3 — контракт M2 -> M3; None — подобрать по силуэту в [k_min, k_max]
    k_min: int = 3
    k_max: int = 5
    feature_sigma_px: float = 1.5
    z_clip: float = 6.0  # защита от единичных выбросов (тени облаков и т.п.)
    majority_size: int = 5  # окно мажоритарного фильтра, пиксели (0 — выключить)
    majority_iters: int = 2
    min_zone_px: int = 25
    max_fit_samples: int = 50_000
    silhouette_samples: int = 4_000
    random_state: int = 0
    order_by: str = "NDRE"  # класс 1 — худший по этому признаку (ТЗ: сортировка по среднему NDRE)


@dataclass
class ZoningResult:
    zones: np.ndarray  # uint8 [H, W]; 0 — нет данных, 1..N — классы
    n_classes: int
    labels: list[str]
    vigor: list[float]  # средний z-score признака order_by по классам (по возрастанию)
    k_requested: int | None = None
    silhouette: dict[int, float] = field(default_factory=dict)


def _choose_k(X: np.ndarray, cfg: ZoningConfig, rng: np.random.Generator) -> tuple[int, dict[int, float]]:
    n = min(len(X), cfg.silhouette_samples)
    sample = X[rng.choice(len(X), size=n, replace=False)]
    scores: dict[int, float] = {}
    for k in range(cfg.k_min, cfg.k_max + 1):
        if k >= len(sample):
            break
        km = KMeans(k, n_init=5, random_state=cfg.random_state).fit(sample)
        if len(np.unique(km.labels_)) < 2:
            continue
        scores[k] = float(silhouette_score(sample, km.labels_))
    if not scores:
        return cfg.k_min, {}
    return max(scores, key=scores.get), scores


def _majority(cls: np.ndarray, k: int, size: int) -> np.ndarray:
    """Мажоритарный фильтр: пиксель получает самый частый класс в окне (только валидные)."""
    valid = cls > 0
    counts = np.stack([
        ndi.uniform_filter(((cls == c) & valid).astype(np.float32), size=size, mode="nearest")
        + 1e-3 * (cls == c)  # при ничьей остаёмся в своём классе
        for c in range(1, k + 1)
    ])
    out = (counts.argmax(axis=0) + 1).astype(np.uint8)
    out[~valid] = 0
    return out


def _merge_small(cls: np.ndarray, min_px: int, max_rounds: int = 5) -> np.ndarray:
    """Пятна меньше min_px пикселей присоединяются к самому частому соседнему классу."""
    if min_px <= 1:
        return cls
    out = cls.copy()
    cross = ndi.generate_binary_structure(2, 1)
    for _ in range(max_rounds):
        changed = False
        for c in np.unique(out[out > 0]):
            lab, n = ndi.label(out == c, structure=cross)
            if n == 0:
                continue
            sizes = ndi.sum(np.ones(lab.shape, dtype=np.int32), lab, index=np.arange(1, n + 1))
            slices = ndi.find_objects(lab)
            for comp_id in np.where(sizes < min_px)[0] + 1:
                sl = slices[comp_id - 1]
                sl = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl)  # +1 px вокруг
                comp = lab[sl] == comp_id
                ring = ndi.binary_dilation(comp, structure=cross) & ~comp
                neigh = out[sl][ring]
                neigh = neigh[neigh > 0]
                if neigh.size:
                    out[sl][comp] = np.bincount(neigh).argmax()
                    changed = True
        if not changed:
            break
    return out


def segment_zones(
    indices: dict[str, np.ndarray], cfg: ZoningConfig | None = None
) -> ZoningResult:
    """Строит растр зон управления по набору индексов (2D float, NaN = нет данных)."""
    cfg = cfg or ZoningConfig()
    smoothed = [smooth_nan(indices[name], cfg.feature_sigma_px) for name in cfg.features]
    valid = np.all([np.isfinite(s) for s in smoothed], axis=0)
    n_valid = int(valid.sum())
    k_floor = cfg.n_classes if cfg.n_classes else cfg.k_max
    if n_valid < max(10 * k_floor, 50):
        raise ValueError(f"слишком мало валидных пикселей для зонирования: {n_valid}")

    z = np.stack([np.clip(zscore(s), -cfg.z_clip, cfg.z_clip) for s in smoothed])  # [F, H, W]
    X = z[:, valid].T  # [N, F]

    rng = np.random.default_rng(cfg.random_state)
    silhouette: dict[int, float] = {}
    if cfg.n_classes is None:
        k, silhouette = _choose_k(X, cfg, rng)
    else:
        k = int(cfg.n_classes)
    k = max(1, min(k, n_valid))

    fit_idx = rng.choice(len(X), size=min(len(X), cfg.max_fit_samples), replace=False)
    km = KMeans(k, n_init=10, random_state=cfg.random_state).fit(X[fit_idx])
    lab = km.predict(X)

    # порядок кластеров: 1 — хуже всех по order_by (по умолчанию NDRE), иначе по среднему всех признаков
    order_idx = cfg.features.index(cfg.order_by) if cfg.order_by in cfg.features else None
    vigor_centroid = (
        km.cluster_centers_[:, order_idx] if order_idx is not None else km.cluster_centers_.mean(axis=1)
    )
    rank = np.empty(k, dtype=np.int64)
    rank[np.argsort(vigor_centroid)] = np.arange(k)

    cls = np.zeros(valid.shape, dtype=np.uint8)
    cls[valid] = (rank[lab] + 1).astype(np.uint8)

    # пространственная регуляризация
    if cfg.majority_size and cfg.majority_size > 1:
        for _ in range(cfg.majority_iters):
            cls = _majority(cls, k, cfg.majority_size)
    cls = _merge_small(cls, cfg.min_zone_px)

    # нумерация подряд 1..N (классы могли исчезнуть при сглаживании)
    present = np.unique(cls[cls > 0])
    remap = np.zeros(k + 1, dtype=np.uint8)
    remap[present] = np.arange(1, len(present) + 1, dtype=np.uint8)
    cls = remap[cls]

    # после сглаживания порядок мог чуть сдвинуться: финально сортируем классы по среднему order_by
    composite = z[order_idx] if order_idx is not None else z.mean(axis=0)
    n_final = len(present)
    means = np.array([float(np.mean(composite[cls == i])) for i in range(1, n_final + 1)])
    if np.any(np.diff(means) < 0):
        new_rank = np.empty(n_final + 1, dtype=np.uint8)
        new_rank[0] = 0
        new_rank[1 + np.argsort(means)] = np.arange(1, n_final + 1, dtype=np.uint8)
        cls = new_rank[cls]
        means = np.sort(means)
    vigor = [float(v) for v in means]
    return ZoningResult(
        zones=cls, n_classes=int(len(present)), labels=class_labels(len(present)),
        vigor=vigor, k_requested=k, silhouette=silhouette,
    )
