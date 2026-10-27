"""Сезонный отчёт модуля 2 (КТ4): маски и графики за весь сезон по уже посчитанным результатам.

Читает out_dir (zones/, indices/, hidden_stress/, stats/) после ``stress_core run`` и пишет в out_dir/season/:

* ``season_zones.png``     — маски зон по датам, контур — скрытый стресс, ⚠ — early_stress_alert;
* ``season_indices.png``   — карты NDVI и NDRE по датам в единой шкале сезона;
* ``season_dynamics.png``  — средние индексы поля и тревоги; очаг относительно поля (опережение NDRE);
                             площади зон по датам;
* ``season_report.json``   — те же цифры для дашборда и отчёта.

Очаг — пиксели скрытого стресса на дате ``focus_date`` (по умолчанию дата с наибольшей площадью скрытого
стресса, после которой есть ещё хотя бы одна дата). Для очага считается, насколько его NDVI и NDRE ниже
медианы поля на каждой дате. «Видимое угнетение» — первая дата, когда NDVI очага ниже поля на
``visible_ndvi_drop`` (10 %); опережение = эта дата минус первая дата, когда очаг отмечен скрытым стрессом.
"""
from __future__ import annotations

import json
from datetime import date as _date
from pathlib import Path

import numpy as np

from .geotiff_io import read_geotiff
from .pipeline import parse_date

ZONE_COLORS = {1: "#e74c3c", 2: "#f39c12", 3: "#2ecc71"}  # как в дашборде M4
ZONE_NAMES = {1: "сильный стресс", 2: "умеренный стресс", 3: "норма"}


def _load_season(out_dir: Path) -> list[dict]:
    rows = []
    for p in sorted((out_dir / "stats").glob("index_stats_*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        d = parse_date(doc["date"])
        stamp = d.replace("-", "_")
        files = doc.get("files") or {}
        idx = files.get("indices") or {}
        try:
            row = {
                "date": d, "stats": doc,
                "zones": read_geotiff(out_dir / files.get("zones", f"zones/zones_{stamp}.tif"))[0][0],
                "hidden": read_geotiff(out_dir / files.get("hidden_stress",
                                                           f"hidden_stress/hidden_stress_{stamp}.tif"))[0][0],
                "NDVI": read_geotiff(out_dir / idx.get("NDVI", f"indices/NDVI_{stamp}.tif"))[0][0].astype(float),
                "NDRE": read_geotiff(out_dir / idx.get("NDRE", f"indices/NDRE_{stamp}.tif"))[0][0].astype(float),
            }
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{d}: нет растра ({exc}); запустите stress_core run без --no-indices") from exc
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"в {out_dir / 'stats'} нет index_stats_*.json — сначала stress_core run")
    return rows


def _days(a: str, b: str) -> int:
    return (_date.fromisoformat(b) - _date.fromisoformat(a)).days


def season_summary(
    out_dir: str | Path, focus_date: str | None = None, visible_ndvi_drop: float = 0.10, flag_share: float = 0.3
) -> tuple[dict, list[dict], np.ndarray]:
    """Цифры отчёта: (report, строки по датам с растрами, маска очага)."""
    out_dir = Path(out_dir)
    rows = _load_season(out_dir)
    dates = [r["date"] for r in rows]
    hidden_px = [int((r["hidden"] == 1).sum()) for r in rows]
    if focus_date is None:
        cand = [(px, i) for i, px in enumerate(hidden_px[:-1]) if px > 0] or [(hidden_px[-1], len(rows) - 1)]
        fi = max(cand)[1]
    else:
        fi = dates.index(parse_date(focus_date))
    focus = rows[fi]["hidden"] == 1

    series = []
    for r in rows:
        rec = {"date": r["date"]}
        for name in ("NDVI", "NDRE"):
            a = r[name]
            med = float(np.nanmedian(a))
            rec[f"field_median_{name.lower()}"] = round(med, 4)
            vals = a[focus & np.isfinite(a)]
            rec[f"focus_mean_{name.lower()}"] = round(float(vals.mean()), 4) if vals.size else None
            rec[f"focus_rel_{name.lower()}"] = (round(float(vals.mean()) / med - 1.0, 4)
                                                if vals.size and med else None)
        n_f = int(focus.sum())
        rec["focus_flagged_share"] = round(float((r["hidden"][focus] == 1).sum()) / n_f, 3) if n_f else None
        st = r["stats"]
        rec["alert"] = bool(st.get("early_stress_alert"))
        rec["reasons"] = (st.get("early_stress") or {}).get("reasons", [])
        rec["hidden_area_ha"] = (st.get("hidden_stress") or {}).get("area_ha")
        rec["hidden_by_rule_ha"] = (st.get("hidden_stress") or {}).get("by_rule_ha")
        rec["mean_ndvi"], rec["mean_ndre"] = st.get("mean_ndvi"), st.get("mean_ndre")
        rec["zones_ha"] = {str(c["class"]): c["area_ha"] for c in (st.get("zones") or {}).get("classes", [])}
        rec["filled_area_ha"] = st.get("filled_area_ha")
        series.append(rec)

    # первая дата непрерывной серии отметок очага, заканчивающейся на focus_date
    first = fi
    while first > 0 and (series[first - 1]["focus_flagged_share"] or 0) >= flag_share:
        first -= 1
    visible = next((s["date"] for s in series[fi:]
                    if s["focus_rel_ndvi"] is not None and s["focus_rel_ndvi"] <= -visible_ndvi_drop), None)
    pixel_ha = None
    st0 = rows[fi]["stats"]
    if st0.get("hidden_stress", {}).get("pixels"):
        pixel_ha = st0["hidden_stress"]["area_ha"] / st0["hidden_stress"]["pixels"]
    report = {
        "dates": dates,
        "alert_dates": [s["date"] for s in series if s["alert"]],
        "focus": {
            "date": dates[fi],
            "pixels": int(focus.sum()),
            "area_ha": round(int(focus.sum()) * pixel_ha, 4) if pixel_ha else None,
            "first_flagged": dates[first],
            "visible_ndvi_date": visible,
            "visible_ndvi_drop": visible_ndvi_drop,
            "lead_days": _days(dates[first], visible) if visible else None,
            "note": "очаг — пиксели скрытого стресса на focus.date; visible_ndvi_date — первая дата, когда NDVI "
                    "очага ниже медианы поля на visible_ndvi_drop; lead_days — опережение детекции по Red Edge",
        },
        "series": series,
    }
    return report, rows, focus


def write_season_report(
    out_dir: str | Path, focus_date: str | None = None, visible_ndvi_drop: float = 0.10
) -> dict:
    """Пишет out_dir/season/*.png и season_report.json; возвращает отчёт."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    out_dir = Path(out_dir)
    report, rows, focus = season_summary(out_dir, focus_date, visible_ndvi_drop)
    dst = out_dir / "season"
    dst.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    ncol = min(4, n)
    nrow = -(-n // ncol)
    label = lambda d: f"{d[8:10]}.{d[5:7]}"  # noqa: E731
    files = {}

    # 1. зоны
    cmap = ListedColormap(["#ffffff", ZONE_COLORS[1], ZONE_COLORS[2], ZONE_COLORS[3]])
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.7 * nrow + 0.6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, r, s in zip(axes.flat, rows, report["series"]):
        z = np.ma.masked_equal(r["zones"], 0)
        ax.imshow(z, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        h = (r["hidden"] == 1).astype(float)
        if h.any():
            ax.contour(h, levels=[0.5], colors="black", linewidths=1.0)
        title = label(r["date"]) + ("  ⚠ ранний стресс" if s["alert"] else "")
        ax.set_title(title, fontsize=10, color="#b03a2e" if s["alert"] else "black")
    fig.legend(handles=[Patch(color=ZONE_COLORS[k], label=f"{k} — {v}") for k, v in ZONE_NAMES.items()]
               + [Patch(facecolor="none", edgecolor="black", label="скрытый стресс (контур)")],
               loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Зоны управления по датам сезона", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97), h_pad=2.0)
    files["zones"] = dst / "season_zones.png"
    fig.savefig(files["zones"], dpi=130)
    plt.close(fig)

    # 2. карты индексов в единой шкале сезона
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n + 1.2, 5.2), squeeze=False)
    for row_i, (name, cm) in enumerate((("NDVI", "RdYlGn"), ("NDRE", "RdYlGn"))):
        allv = np.concatenate([r[name][np.isfinite(r[name])] for r in rows])
        vmin, vmax = np.percentile(allv, [2, 98])
        for j, r in enumerate(rows):
            ax = axes[row_i, j]
            im = ax.imshow(r[name], cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
            if name == "NDRE":
                h = (r["hidden"] == 1).astype(float)
                if h.any():
                    ax.contour(h, levels=[0.5], colors="black", linewidths=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_i == 0:
                ax.set_title(label(r["date"]), fontsize=9)
            if j == 0:
                ax.set_ylabel(name, fontsize=10)
        fig.colorbar(im, ax=axes[row_i, :].tolist(), shrink=0.85, pad=0.01)
    fig.suptitle("NDVI и NDRE за сезон (единая шкала; контур — скрытый стресс)", fontsize=11)
    files["indices"] = dst / "season_indices.png"
    fig.savefig(files["indices"], dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 3. динамика
    ser = report["series"]
    x = np.arange(n)
    xt = [label(d) for d in report["dates"]]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    ax = axes[0]
    ax.plot(x, [s["mean_ndvi"] for s in ser], "o-", color="#1a9850", label="NDVI (среднее по полю)")
    ax.plot(x, [s["mean_ndre"] for s in ser], "s-", color="#7b3294", label="NDRE (среднее по полю)")
    for i, s in enumerate(ser):
        if s["alert"]:
            ax.axvspan(i - 0.4, i + 0.4, color="#e74c3c", alpha=0.12, lw=0)
    ax.set_ylabel("индекс")
    ax.set_title("Поле целиком; красная заливка — early_stress_alert", fontsize=10)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    f = report["focus"]
    rel = lambda k: [None if s[k] is None else 100 * s[k] for s in ser]  # noqa: E731
    ax.plot(x, rel("focus_rel_ndvi"), "o-", color="#1a9850", label="NDVI очага относительно поля")
    ax.plot(x, rel("focus_rel_ndre"), "s-", color="#7b3294", label="NDRE очага относительно поля")
    ax.axhline(0, color="grey", lw=0.8)
    ax.axhline(-100 * visible_ndvi_drop, color="#1a9850", lw=0.8, ls=":")
    i_first = report["dates"].index(f["first_flagged"])
    ax.axvline(i_first, color="#7b3294", ls="--", lw=1.2, label="очаг найден по Red Edge")
    if f["visible_ndvi_date"]:
        i_vis = report["dates"].index(f["visible_ndvi_date"])
        ax.axvline(i_vis, color="#1a9850", ls="--", lw=1.2,
                   label=f"виден по NDVI (−{int(100 * visible_ndvi_drop)} % к полю)")
    lead = f"опережение {f['lead_days']} дн." if f["lead_days"] is not None else "NDVI не упал до порога"
    area = f"{f['area_ha']:.1f} га" if f["area_ha"] else f"{f['pixels']} px"
    ax.set_title(f"Очаг {label(f['date'])} ({area}): {lead}", fontsize=10)
    ax.set_ylabel("% к медиане поля")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)

    ax = axes[2]
    bottom = np.zeros(n)
    for k in (1, 2, 3):
        vals = np.array([s["zones_ha"].get(str(k)) or 0.0 for s in ser])
        ax.bar(x, vals, bottom=bottom, color=ZONE_COLORS[k], label=f"{k} — {ZONE_NAMES[k]}")
        bottom += vals
    ax2 = ax.twinx()
    ax2.plot(x, [s["hidden_area_ha"] or 0 for s in ser], "k^-", label="скрытый стресс, га")
    ax2.set_ylabel("скрытый стресс, га")
    ax2.set_ylim(bottom=0)
    ax.set_ylabel("площадь зон, га")
    ax.set_title("Площади зон и скрытого стресса", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4,
              frameon=False)
    ax.set_xticks(x, xt)
    fig.tight_layout()
    files["dynamics"] = dst / "season_dynamics.png"
    fig.savefig(files["dynamics"], dpi=130)
    plt.close(fig)

    report["files"] = {k: str(v.relative_to(out_dir)).replace("\\", "/") for k, v in files.items()}
    (dst / "season_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
