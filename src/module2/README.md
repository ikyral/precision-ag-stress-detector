# Модуль 2. Аналитическое ядро: спектральные индексы и сегментация стресса

Вход: многоканальные GeoTIFF-сцены (`scene_YYYY_MM_DD.tif`, отражение, по одной на дату).
Выход: маски зон `zones_YYYY_MM_DD.tif` (0 фон, 1 сильный стресс, 2 умеренный, 3 норма), файлы индексов,
`index_stats_YYYY_MM_DD.json` на каждую дату (для дашборда M4) и сводный `indices_summary.json`.

## Запуск

Команды выполняются из папки `src/module2` (рядом с `stress_core/`).

```bash
pip install numpy scipy scikit-learn tifffile      # обязательные
pip install spyndex                                 # опционально (бэкенд spyndex и сверка формул)

# серия сцен -> зоны, статистика, индексы; -o data кладёт результаты в data/zones, data/stats, data/indices
python -m stress_core run ../../data/scenes -o ../../data --bands s2-10

# одна сцена без даты в имени (как sample_scene.tif): дату передаём явно
python -m stress_core run ../../data/samples/sample_scene.tif -o out --bands s2-10 --date 2026-08-07

# только индексы одной сцены: NDVI.tif, NDRE.tif, MSAVI.tif, NDMI.tif
python -m stress_core indices ../../data/samples/sample_scene.tif -o out_indices --bands s2-10

# демо на синтетическом поле (3 даты)
python -m stress_core demo -o demo_out
```

`--bands s2-10` означает порядок каналов `B,G,R,RE1,RE2,RE3,N,N2,S1,S2` (формат `sample_scene.tif`, см. `data/README.md`).
Для другого файла перечислите имена из каталога ASI через запятую (по умолчанию `R,RE1,N,S1`). `--classes 3|4|5|auto`, по умолчанию 3 (контракт M2 -> M3).

Из Python:

```python
from stress_core import analyze_field, run_pipeline, ZoningConfig
rec = analyze_field("scene_2026_08_07.tif", "data", band_order=("B","G","R","RE1","RE2","RE3","N","N2","S1","S2"))
summary = run_pipeline(["data/scenes"], "data", band_order=(...))   # вся серия, с динамикой между датами
```

Тесты: `python -m unittest discover -s tests -v` (из `src/module2`).

## Контракт входа

- GeoTIFF, каналы идут в порядке `--bands`. Целочисленные файлы делятся на 10000 (меняется через `--scale`), float считается отражением 0..1.
- Пиксели вне поля — nodata (0) во всех каналах, облака и тени замаскированы **до** модуля (NaN или nodata). Валиден пиксель, где все нужные каналы конечны и `0 < отражение <= 1.5`.
- Все каналы на одной сетке 10 м (ресемплинг — задача M1), CRS проекционная в метрах (UTM), иначе `area_ha` будет `null`.
- Серия сцен в одном запуске даёт временную динамику; одна сцена даёт только пространственные признаки.

## Контракт выхода (`<out>/`)

| Файл | Содержимое |
|---|---|
| `zones/zones_YYYY_MM_DD.tif` | uint8: `0` нет данных, `1` сильный стресс (худший по NDRE), `2` умеренный, `3` норма. Для M3 нужны 3 класса (по умолчанию) |
| `stats/index_stats_YYYY_MM_DD.json` | статистика даты и `early_stress_alert` (см. ниже) |
| `indices/NDVI_YYYY_MM_DD.tif` (+ `NDRE`, `MSAVI`, `NDMI`) | одноканальные float32, NaN = нет данных |
| `hidden_stress/hidden_stress_YYYY_MM_DD.tif` | uint8: `1` скрытый стресс, `0` нет, `255` нет данных |
| `indices_summary.json` | то же по всем датам, плюс конфигурация запуска |

Геопривязка (CRS, шаг, начало координат) копируется из входной сцены.

**`index_stats_YYYY_MM_DD.json`.** Поля для дашборда: `date`, `mean_ndvi`, `mean_ndre`, `mean_msavi`, `mean_ndmi`,
`early_stress_alert` (bool), `early_stress_area_ha`, `early_stress_percent`, `field_area_ha`. Дополнительно: `grid`, `valid_pixels`,
`indices.<имя>` (`n, mean, std, min, max, p5…p95`), `zones` (`n_classes`, `classes[]` с `class, label, pixels, fraction, area_ha, mean{индекс}`),
`hidden_stress`, `early_stress` (`alert`, `reasons`, `ndre_drop`, `ndvi_drop`), `files`. Если сцена не обработалась, в `indices_summary.json` у даты будет поле `error`, остальные даты считаются дальше.

## Как это работает

1. **Индексы** (`indices.py`): NDVI, NDRE, MSAVI2 (в файлах и ключах `MSAVI` / `mean_msavi`), NDMI по формулам каталога Awesome Spectral Indices; значения сверены со spyndex тестом `test_matches_spyndex`.
2. **Скрытый стресс** (`early_stress.py`): NDRE и NDVI сглаживаются (σ=1 px), переводятся в робастный z-score относительно самого поля. Флаг — NDRE заметно ниже нормы (`z <= -1.5`) при ещё нормальном NDVI (`z > -1.0`). Если есть предыдущая дата, добавляется временное условие: падение аномалии NDRE между датами при почти неизменной аномалии NDVI. Мелкие пятна (< 12 px) убираются.
3. **`early_stress_alert`**: `true`, если скрытый стресс занимает не менее 1% поля, либо (при наличии прошлой даты) средний NDRE упал минимум на 0.02 и относительно быстрее NDVI в 1.5 раза и более. Пороги в `EarlyStressConfig`.
4. **Зоны** (`zoning.py`): признаки (NDVI, NDRE, MSAVI2, NDMI) сглаживаются и стандартизируются, KMeans (k задан или по силуэту в 3–5), классы сортируются по среднему NDRE (1 — самый низкий), затем мажоритарный фильтр и слияние пятен < 25 px с соседом.

## Ограничения и допущения

- **Зоны относительные.** Класс 1 — «хуже остального поля на эту дату», а не абсолютный порог. На здоровом поле алгоритм всё равно выделит классы из-за неоднородности почвы, поэтому подписи читайте вместе со средними индексами класса. Смысл номера класса между датами не гарантирован.
- **Скрытый стресс тоже относительный:** если угнетена больше половины поля, медиана смещается и метод занижает очаги.
- Пороги (`EarlyStressConfig`, в том числе пороги тревоги) подобраны на синтетике и одной реальной сцене, на полевой разметке **не калибровались**. Динамика между датами проверена только на синтетических сериях.
- Края поля (смешанные пиксели) могут попадать в класс 1.
- Xarray не используется, расчёт на NumPy.
- Бэкенд `spyndex` (`--backend spyndex`) проверен тестом на машине автора; в окружении без пакета тест пропускается.
