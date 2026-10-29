# Данные и форматы обмена

Цепочка: М1 (`scenes/`) → М2 (`zones/`, `stats/`, `indices/`, `hidden_stress/`) → М3 (`prescriptions/`) →
М1 (`overlays/` для кабинета) → М4 (читает всё это). Целиком её запускает `main_pipeline.py` из корня
репозитория, через Python из `.venv` (Windows: `.venv\Scripts\python.exe`):

```bash
.venv/bin/python main_pipeline.py --skip-fetch                          # по сценам, которые уже лежат в data/scenes
.venv/bin/python main_pipeline.py --start 2026-06-29 --end 2026-08-30   # сначала докачать новые сцены (ключи CDSE в .env)
```

## Раскладка `data/` после конвейера

| Путь | Кто пишет | В git | Содержимое |
|---|---|---|---|
| `samples/` | М1 (КТ1) | да | эталоны: `sample_scene.tif`, `sample_zones.tif`, `field_boundary.geojson` (см. ниже) |
| `scenes/scene_YYYY_MM_DD.tif` | М1 `s2fetch` | да | серия сцен сезона 2026 (8 дат) |
| `scenes/scenes_manifest.json` | М1 | да | облачность и решение по каждой найденной дате |
| `scenes/preview/` | М1 | нет | PNG для глазной проверки облаков |
| `zones/zones_YYYY_MM_DD.tif` | М2 | нет | маска зон, uint8: 0 вне поля, 1 сильный стресс, 2 умеренный, 3 норма; облачные пиксели поля — класс ближайшей зоны |
| `stats/index_stats_YYYY_MM_DD.json` | М2 | нет | статистика даты, `early_stress_alert`, `indices.<NDVI\|NDRE\|MSAVI2\|NDMI>.p5…p95`, `filled_area_ha` |
| `indices/{NDVI,NDRE,MSAVI,NDMI}_YYYY_MM_DD.tif` | М2 | нет | float32, NaN — вне поля или облако |
| `hidden_stress/hidden_stress_YYYY_MM_DD.tif` | М2 | нет | uint8: 1 скрытый стресс, 0 нет, 255 нет данных |
| `indices_summary.json` | М2 | нет | сводка М2 по всем датам |
| `prescriptions/prescription_YYYY_MM_DD.geojson`, `shapefile_YYYY_MM_DD.zip`, `metrics_YYYY_MM_DD.json` | М3 | нет | карта-предписание (EPSG:4326), Shapefile для трактора (EPSG:32637), экономика |
| `prescriptions/season_summary.json` | М3 / `main_pipeline` | нет | свод сезона, схема `vra.batch.run_series` (`area_stable`, `records`, `errors`) |
| `overlays/{NDVI,NDRE,MSAVI,NDMI,RGB,HIDDEN}_YYYY_MM_DD.png` | М1 `s2fetch.overlays` | нет | окрашенные слои для карты М4 |
| `overlays/meta.json` | М1 | нет | границы и шкалы слоёв (контракт ниже) |
| `overlays/preview_all.png` | М1 | нет | контактный лист «даты × слои» |
| `overlays/overlay_check.html` | М1 `tools/overlay_map.py` | нет | карта для проверки совмещения в браузере |
| `field_boundary.geojson` | `main_pipeline` | нет | копия контура для М4, только если `--data-dir` не `data/` |
| `pipeline_state.json` | `main_pipeline` | нет | sha256 сцен и масок зон, по которым считали М2 и М3 |
| `pipeline_report.json` | `main_pipeline` | нет | отчёт последнего запуска (схема 1.1): этапы, даты, `errors`, `warnings`, `info` (предупреждения М2, ложные по sha256, с пояснением), проверки (в т. ч. `crs_audit`: CRS и площади выходов М3 против сетки сцен), время. М4 показывает `errors`: даты и этапы с ошибкой. Без пакетов М2–М4 запуск останавливается до этапов и отчёт не перезаписывает |

В git лежат только `samples/`, `scenes/*.tif`, `scenes/scenes_manifest.json` и этот файл (см. `.gitignore`).
Всё остальное воспроизводится командой `.venv/bin/python main_pipeline.py --skip-fetch` примерно за 4 с.

## М1 → М2: `scenes/scene_YYYY_MM_DD.tif`

- GeoTIFF float32, отражение 0..1, 10 каналов с описаниями `B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12`;
- `0` во всех каналах — вне контура поля (`nodata=0`), `NaN` во всех каналах — облако/тень внутри контура;
- EPSG:32637, 10 м, сетка 104×104 как у `samples/sample_scene.tif`; маска поля 7960 px одна на все даты.

Подробности — `src/module1/README.md`, контракт входа М2 — `src/module2/README.md`.

## М1 → М4: `overlays/meta.json`

PNG строятся на сетке **EPSG:3857** (Leaflet растягивает `ImageOverlay` линейно в веб-Меркаторе), около 8
выходных пикселей на пиксель 10 м, метод nearest. Размер и экстент одни на все даты и слои. Поля, которые
читает М4 (`src/module4/app.py`), — в его формате.

| Поле | Читает М4 | Значение |
|---|---|---|
| `bounds` | да | `[[south, west], [north, east]]` — lat/lon углов прямоугольника EPSG:3857, одни на все PNG |
| `index_ranges` | да | `{NDVI, NDRE, MSAVI, NDMI: [lo, hi]}` — шкала, по которой окрашены PNG; формула `season_ranges` М4: по всем `stats/index_stats_*.json` `lo=min(p5)`, `hi=max(p95)`, `pad=0.05·(hi−lo)`, `(round(lo−pad,3), round(hi+pad,3))` |
| `field_name` | да | **не пишется**: имя поля берётся из `field_boundary.geojson` |
| `colormaps` | нет | палитры `CMAP_VEG` (NDVI, NDRE, MSAVI) и `CMAP_WATER` (NDMI) из `app.py`: `[[доля, [r, g, b]], ...]` |
| `alpha` | нет | 235 — альфа индексов и HIDDEN внутри поля (как `colorize` в М4). PNG М4 накладывает с `opacity=1` (альфа уже в пикселях); `opacity=0.85` — только у `indices/*.tif`, когда PNG нет |
| `nodata_mode` | легенда | облако/тень внутри поля: `gray` (по умолчанию) или `transparent`; вне поля всегда alpha 0. При `transparent` М4 не рисует в легенде квадрат «нет данных» |
| `nodata_color`, `nodata_meaning` | `nodata_color` — запасной цвет легенды | `gray` → `[128,128,128,150]`, `transparent` → `[0,0,0,0]`; М4 берёт его, если нет `nodata_css` (схема 1.0). `nodata_meaning` — длинное пояснение, не для легенды, М4 не читает |
| `nodata_label`, `nodata_css` | легенда (1.1) | короткая подпись «Нет данных (облако/тень)» и CSS того же цвета: `rgba(128, 128, 128, 0.588)` (`transparent` → `rgba(0, 0, 0, 0)`) |
| `rgb` | нет | true color B04, B03, B02, alpha 255. Растяжка одна на три канала и на сезон: `min_reflectance`/`max_reflectance` — перцентили `percentiles` (2, 98) отражения всех трёх каналов по пикселям поля всех дат (`season_dates`); `gamma` 1.3; `formula` — как считается пиксель. Превью М1 (`scenes/preview`) растягивается по-старому (0…0.18, гамма 1.6) |
| `rgb.label`, `rgb.legend_text` | `legend_text` — легенда (1.1); `label` М4 не читает (подпись радио — из `LAYERS` `app.py`) | «Снимок Sentinel-2 (RGB)» и «Sentinel-2 L2A, true color (B04, B03, B02), усиление контраста» (подписи входят в `inputs_sha256`: новая подпись перерисовывает слои) |
| `hidden` | `hidden.color` — запасной цвет легенды | слой HIDDEN: 1 — цвет `[231,76,60,235]`, 0 — прозрачно, 255 в поле — `nodata_color` |
| `hidden.label`, `hidden.css` | легенда (1.1) | «Скрытый стресс (М2)» и `rgba(231, 76, 60, 0.922)` |
| `schema_version`, `generated_at`, `generator` | нет | версия схемы (`1.1`), время, кто записал |
| `render_crs`, `render_grid` | нет | `EPSG:3857`; размер PNG, шаг (м), `bounds_3857`, `oversample`, метод |
| `source_grid` | нет | CRS, transform и размер сетки сцен |
| `dates` | нет | даты со слоями, `YYYY_MM_DD` |
| `layers` | нет | `{дата: {слой: имя PNG}}` |
| `sources` | нет | `{дата: {scene, zones, indices, hidden_stress}}` — входные файлы (пути от `data/`) |
| `inputs_sha256` | нет | отпечаток входов и параметров: если он не изменился, PNG не перерисовываются |

Шкала зависит от всех дат, поэтому при новой дате перерисовываются все PNG. В кабинете 7 слоёв (`LAYERS` в
`src/module4/app.py`): карта предписаний VRA (полигоны М3), RGB, HIDDEN и четыре индекса. Для индексов М4 берёт PNG
вместо `indices/*.tif`, если есть `overlays/{INDEX}_{YYYY_MM_DD}.png` и `meta.bounds`, а шкалу легенды — из
`meta.index_ranges`; без PNG рисует `indices/*.tif` с `opacity=0.85`. RGB и HIDDEN есть только готовыми PNG
(`PNG_ONLY_LAYERS`): без них кабинет просит запустить `main_pipeline.py`. Подписи этих слоёв в легенде — из
`meta.json`, в `app.py` только запасные.

Легенда (схема 1.1, только добавления к 1.0): `*_css` — тот же цвет, которым закрашены пиксели PNG
(`rgba(r, g, b, alpha/255)`, альфа с 3 знаками), без `opacity` слоя: для PNG М4 ставит `opacity=1`. Цвета и
CSS берутся из одних констант `s2fetch.overlays` (`HIDDEN_COLOR`, `NODATA_GRAY`), тест сверяет их с пикселями PNG.

## Эталоны `samples/`

### sample_scene.tif
- Каналы (10 шт.): 1:B02, 2:B03, 3:B04, 4:B05, 5:B06, 6:B07, 7:B08, 8:B8A, 9:B11, 10:B12
- Соответствие для spyndex: B, G, R, RE1, RE2, RE3, N, N2, S1, S2
- Тип: uint16, значения = отражение x 10000 (смещение +1000 из обработки N05.xx уже вычтено)
- nodata = 0 (вне контура поля)
- Разрешение: 10 м, CRS: EPSG:32637 (UTM 37N)
- Источник: Sentinel-2 L2A, плитка T37TEN, дата 2026-08-07, облачность 0,5%
- Размер: 104 x 104 пикселя (круг около 79 га, радиус ~500 м)

### sample_zones.tif
- uint8: 0 вне поля, 1 сильный стресс, 2 умеренный, 3 норма
- Та же сетка и CRS, что у sample_scene.tif
- ВНИМАНИЕ: тестовая маска по терцилям NDVI (классы ~по 33%), не реальный стресс
- `main_pipeline` передаёт её в М3 как эталон контура (`reference_mask_path`): по ней считается `coverage`

### field_boundary.geojson
- Контур поля, EPSG:4326; по нему М1 строит сетку и маску поля, М4 рисует границу.
