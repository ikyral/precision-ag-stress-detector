# Precision Ag Stress Detector

Учебный проект: обнаружение стресса растений по снимкам Sentinel-2 и генерация карт дифференцированного внесения (VRA).

## Модули
1. Спутниковый пайплайн и гео-бэкенд (`src/module1`)
2. Спектральные индексы и сегментация (`src/module2`)
3. Карты-предписания VRA и экономика (`src/module3`)
4. Веб-кабинет агронома (`src/module4`)

## Формат обмена данными
См. `data/README.md`.

## Настройка
Скопируйте `.env.example` в `.env` и впишите ключи Copernicus S3 (нужны только для скачивания сцен).

Окружение — `.venv` в корне, зависимости всех модулей:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r src/module1/requirements.txt -r src/module2/requirements.txt \
    -r src/module3/requirements.txt -r src/module4/requirements.txt
```

## Запуск всего конвейера (`main_pipeline.py`)

М1 (сцены Sentinel-2) → М2 (индексы, зоны) → М3 (предписания VRA) → слои PNG для кабинета М4. Обрабатываются
только новые даты; повторный запуск без новых снимков меняет только `data/pipeline_report.json`.

Linux / macOS:

```bash
.venv/bin/python main_pipeline.py --skip-fetch                          # без ключей: сцены из data/scenes
.venv/bin/python main_pipeline.py --start 2026-06-29 --end 2026-08-30   # докачать новые сцены и обработать
.venv/bin/streamlit run src/module4/app.py                               # кабинет
```

Windows (PowerShell):

```powershell
.venv\Scripts\python.exe main_pipeline.py --skip-fetch
.venv\Scripts\python.exe main_pipeline.py --start 2026-06-29 --end 2026-08-30
.venv\Scripts\streamlit.exe run src\module4\app.py
```

Параметры: `--data-dir` (по умолчанию `data/`), `--no-weekly` (по умолчанию одна сцена на неделю), `--force`
(перекачать сцены), `--rebuild` (М2, М3 и слои по всем датам заново), нормы М3 `--rates 1=150,2=80,3=0`,
`--base-rate 150`, `--price 45`, `--min-area 0.05`, `--nodata-mode gray|transparent`. Код выхода: 0 — ок,
1 — ошибки обработки, 2 — данные есть, но проверки не прошли. Подробности — `docs/module1_kt3.md`.

## Лицензия и данные
Код: MIT. Contains modified Copernicus Sentinel data 2026.
