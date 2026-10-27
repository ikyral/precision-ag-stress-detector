"""Выбор сцен: не больше одной на неделю (самая чистая)."""
from __future__ import annotations

import datetime as dt


def iso_week(date: str) -> str:
    y, w, _ = dt.date.fromisoformat(date).isocalendar()
    return f"{y}-W{w:02d}"


def pick_weekly(clean: dict[str, float]) -> dict[str, str]:
    """{дата: доля облаков} чистых сцен -> {неделя: выбранная дата}.

    На неделе выбирается дата с наименьшей долей облаков; при равенстве — более ранняя.
    """
    best: dict[str, str] = {}
    for date in sorted(clean):
        wk = iso_week(date)
        if wk not in best or clean[date] < clean[best[wk]]:
            best[wk] = date
    return best


def weeks_between(start: str, end: str) -> list[str]:
    """Все ISO-недели интервала по порядку (для показа пропусков)."""
    d, last = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    weeks: list[str] = []
    while d <= last:
        wk = iso_week(d.isoformat())
        if not weeks or weeks[-1] != wk:
            weeks.append(wk)
        d += dt.timedelta(days=1)
    return weeks
