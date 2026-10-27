"""Поиск сцен в CDSE STAC: фильтр плитки, одна сцена на дату, пара L1C для s2cloudless."""
from __future__ import annotations

import os
from dataclasses import dataclass

STAC_URL = os.environ.get("CDSE_STAC_URL", "https://stac.dataspace.copernicus.eu/v1")
L2A = "sentinel-2-l2a"
L1C = "sentinel-2-l1c"


@dataclass(frozen=True)
class SceneId:
    """Разбор имени продукта: S2B_MSIL2A_20260807T081609_N0512_R121_T37TEN_20260807T110039."""

    id: str
    mission: str      # S2A / S2B / S2C
    level: str        # MSIL2A / MSIL1C
    sensing: str      # 20260807T081609
    baseline: int     # 512 для N0512
    orbit: str        # R121
    tile: str         # T37TEN
    processed: str    # 20260807T110039

    @property
    def date(self) -> str:
        s = self.sensing
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def parse_id(item_id: str) -> SceneId:
    name = item_id.removesuffix(".SAFE")
    parts = name.split("_")
    if len(parts) != 7 or not parts[3].startswith("N"):
        raise ValueError(f"Непонятное имя продукта Sentinel-2: {item_id}")
    return SceneId(item_id, parts[0], parts[1], parts[2], int(parts[3][1:]),
                   parts[4], parts[5], parts[6])


def boa_offset(item_id: str) -> int:
    """Смещение +1000 есть у обработки с версии 04.00 (N0400 и новее), и у L2A, и у L1C."""
    return 1000 if parse_id(item_id).baseline >= 400 else 0


def _newest(ids: list[SceneId]) -> SceneId:
    # самая новая обработка: сначала версия, затем время обработки
    return max(ids, key=lambda s: (s.baseline, s.processed))


def one_per_date(item_ids: list[str], tile: str) -> dict[str, str]:
    """{дата: id}. Только заданная плитка MGRS; на дату — самая новая обработка."""
    by_date: dict[str, list[SceneId]] = {}
    for i in item_ids:
        s = parse_id(i)
        if s.tile == tile:
            by_date.setdefault(s.date, []).append(s)
    return {d: _newest(v).id for d, v in sorted(by_date.items())}


def match_l1c(l2a_id: str, l1c_ids: list[str]) -> str | None:
    """L1C той же съёмки (спутник, время съёмки, орбита, плитка); самая новая обработка."""
    a = parse_id(l2a_id)
    same = [s for s in map(parse_id, l1c_ids)
            if (s.mission, s.sensing, s.orbit, s.tile) == (a.mission, a.sensing, a.orbit, a.tile)]
    return _newest(same).id if same else None


def open_client(url: str = STAC_URL):
    from pystac_client import Client
    from pystac_client.stac_api_io import StacApiIO
    from urllib3 import Retry

    # Повторять запросы с паузой при 429 и 5xx (как в КТ1)
    retry = Retry(total=6, backoff_factor=3, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=None)
    return Client.open(url, stac_io=StacApiIO(max_retries=retry))


def search(client, collection: str, bbox, start: str, end: str) -> dict:
    """{id: item} за интервал дат [start, end] включительно."""
    items = client.search(collections=[collection], bbox=bbox,
                          datetime=f"{start}T00:00:00Z/{end}T23:59:59Z", limit=100).items()
    return {it.id: it for it in items}
