# Модуль 1 на защите: сценарий и аварийная инструкция

Все цифры ниже взяты из файлов репозитория. Источник указан в скобках: `manifest` — `data/scenes/scenes_manifest.json`,
`season` — `docs/module1_season_2026.md`, `kt3` — `docs/module1_kt3.md`, `kt4` — `docs/module1_kt4.md`,
`meta` — `data/overlays/meta.json`.

## Сценарий части М1 (2–3 минуты)

**1. Откуда данные (≈30 с).** Снимки Sentinel-2 L2A берём из Copernicus Data Space (CDSE): поиск через STAC,
чтение каналов окном прямо из S3, без скачивания целых плиток. Поле — поливной круг 79,6 га на плитке
**T37TEN** (season). За сезон найдено 46 сцен L2A и 46 L1C. Чистых недель, где облаков в поле ≤ 10 %, — 16 из 23
(season). Для серии взято окно 2026-06-29 … 2026-08-30 (manifest): 18 дат съёмки. На каждую ISO-неделю
оставлена самая чистая сцена — **8 дат, 07-03 … 08-24**. Неделя W30 целиком в облаках (season, manifest).

**2. Как отсеяны облака (≈45 с).** Маска облаков в поле — объединение трёх источников, плюс расширение на 2 пикселя (manifest):
- **SCL** — классы 3, 8, 9, 10: тень, облака, перистые;
- **s2cloudless** по L1C той же съёмки: вероятность > 0,4, `average_over=22`;
- **ручная маска** для 18.07 (`src/module1/manual_masks.geojson`, 4 полигона, 162 px; season). На юго-востоке
  поля тонкое облако ~81 px. SCL считает его растительностью (класс 4), MSK_CLDPRB — 0, s2cloudless даёт
  максимум 0,20 при пороге 0,4. Его тень лежит в 720 м по азимуту ≈ 328°, высота облака ≈ 1,4 км (season).
  С маской на 18.07 в поле 5,18 % NaN (412 px, manifest). Дальше М2 заполняет их классом ближайшей зоны.

Порог отбора даты — 10 % облаков в поле (manifest). Внутри контура облако — это NaN, вне контура — 0.

**3. Одна сетка на все модули (≈20 с).** EPSG:32637, шаг 10 м, 104×104 пикселя, маска поля 7960 px (manifest).
Эту сетку задаёт эталон `samples/sample_scene.tif`. В `main_pipeline` 56 растров сцен, зон, индексов и скрытого
стресса сверяются с ним по CRS, transform и размеру (kt3). Со сквозной проверкой КТ4 сверяются ещё Shapefile М3
(EPSG:32637), GeoJSON (CRS84) и площади: 79,6 га совпадают у М2, М3 и масок зон на всех 8 датах (kt4).

**4. PNG-слои для кабинета (≈20 с).** Leaflet растягивает картинку линейно в веб-Меркаторе, поэтому слои
NDVI, NDRE, MSAVI, NDMI, RGB и HIDDEN рисуются сразу в EPSG:3857: PNG 846×849, шаг 1,843 м, 8 выходных пикселей
на пиксель 10 м (meta, kt3). Всего 48 PNG на 8 дат (`main_pipeline`, `png_count`). Тест проверяет, что центр и
углы каждого из 7960 пикселей поля попадают в свою клетку на PNG (kt3). Шкалы и цвета совпадают с легендой М4,
подписи легенды лежат в `meta.json` (схема 1.1).

**5. `main_pipeline`: от снимка до кабинета (≈30 с).** Одна команда проводит снимок через все модули:
М1 → М2 (индексы, зоны) → М3 (предписания, Shapefile, экономика) → слои → проверки → `pipeline_report.json`.
- Реальный прогон со скачиванием занял 207 с, из них М1 — 205 с. Все 8 сцен побайтно совпали с `data/scenes` (kt3).
- С `--skip-fetch` по чистой папке — около 4 с (3,96 с при проверке КТ4). Повторный запуск — меньше 1 с и ни
  одного изменённого файла (kt3).
- Если сцену подменили, М2 переобрабатывает только эту дату и более поздние: подмена 08-07 — 2,2 с (kt4).

Что показать вживую: `python main_pipeline.py --skip-fetch` (строки этапов и `crs_audit: ok`), затем кабинет
М4. Слой NDVI, дата 18.07: серые пятна «нет данных» — облако на юго-востоке и его тень на севере поля (в 720 м на ССЗ).
Затем слой «Карта предписаний VRA».

## Аварийная инструкция: поднять демо с нуля

Для демо (`--skip-fetch`) нужны **только зависимости М2–М4**. Сцены уже лежат в git (`data/scenes/*.tif`).
Зависимости М1 (`pystac-client`, `s2cloudless`, `python-dotenv`) и ключи CDSE нужны только для скачивания новых
снимков (последний раздел). Python 3.11+.

### Linux / macOS

```bash
git clone https://github.com/ikyral/precision-ag-stress-detector.git
cd precision-ag-stress-detector
python3 -m venv .venv
.venv/bin/python -m pip install -r src/module2/requirements.txt -r src/module3/requirements.txt \
    -r src/module4/requirements.txt
.venv/bin/python main_pipeline.py --skip-fetch          # ~4 с, «код выхода 0»
.venv/bin/streamlit run src/module4/app.py              # кабинет: http://localhost:8501
```

### Windows (PowerShell)

```powershell
git clone https://github.com/ikyral/precision-ag-stress-detector.git
cd precision-ag-stress-detector
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r src\module2\requirements.txt -r src\module3\requirements.txt `
    -r src\module4\requirements.txt
.venv\Scripts\python.exe main_pipeline.py --skip-fetch
.venv\Scripts\streamlit.exe run src\module4\app.py
```

Кабинет сам находит папку `data/` вверх от `app.py`. Другую папку можно задать переменной `PAS_DATA_DIR`
(`export PAS_DATA_DIR=/путь/data`, в PowerShell — `$env:PAS_DATA_DIR = "C:\путь\data"`) и выбрать её
в боковой панели.

### Если нет сети

- `pip install` без сети не пройдёт. Ставьте зависимости заранее; на крайний случай подойдёт уже настроенный
  ноутбук или копия папки вместе с `.venv`.
- `main_pipeline.py --skip-fetch` сеть не использует, всё считается по `data/scenes`.
- В кабинете **не загрузится подложка Esri**: спутникового фона под полем не будет. Слои PNG, контур поля,
  полигоны предписаний, графики и экономика работают: они читаются из `data/`.
- Без сети не скачать новые снимки. Для демо это и не нужно.

### Если что-то пошло не так

| Симптом | Что делать |
|---|---|
| `код выхода 1`, `ОШИБКА … [M2]` | смотреть `data/pipeline_report.json` → `errors`; битую сцену вернуть: `git checkout -- data/scenes` |
| `код выхода 2` (проверки) | `pipeline_report.json` → `checks`: какая проверка `ok: false`; `crs_audit.errors` перечисляет даты и причину |
| кабинет «Нет данных» | запустить `main_pipeline.py --skip-fetch`; проверить, что `data/stats/` не пуст, или задать `PAS_DATA_DIR` |
| занят порт 8501 | `streamlit run src/module4/app.py --server.port 8502` |
| результаты «странные» после правок | `main_pipeline.py --skip-fetch --rebuild`: вся серия заново (~4 с) |

### Чек-лист перед защитой (накануне и за час)

- [ ] `git status` чист, ветка — `main` после слияния всех PR; `git status data/scenes data/samples` пуст.
- [ ] Тесты М1: `cd src/module1 && ../../.venv/bin/python -m unittest discover -s tests` — OK.
- [ ] `main_pipeline.py --skip-fetch` — «проверки: ok», «код выхода 0», строка `crs_audit: ok, дат 8`.
- [ ] `src/module1/tools/check_m4.py data --expect-dates 8` — `"ok": true` (все даты × слои кабинета без ошибок).
- [ ] Кабинет открыт заранее, выбрана дата 18.07, слой NDVI. Подложка Esri загрузилась (или известно, что сети нет).
- [ ] Порт свободен; ноутбук на зарядке; вторая машина или копия `.venv` — на крайний случай.

### Скачивание новых снимков (только М1, не для демо)

```bash
.venv/bin/python -m pip install -r src/module1/requirements.txt   # pystac-client, s2cloudless, python-dotenv
cp .env.example .env                                              # ключи CDSE S3: CDSE_S3_ACCESS_KEY, CDSE_S3_SECRET_KEY
.venv/bin/python main_pipeline.py --start 2026-06-29 --end 2026-08-30
```

Без пакетов М1 или без ключей этап М1 завершается одной строкой-подсказкой с кодом 1. Остальные этапы идут по
уже скачанным сценам (kt3).
