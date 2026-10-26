# Модуль 4 — веб-кабинет агронома (Precision Ag Stress Detector)

Дашборд на Streamlit: KPI экономии, спутниковая карта Folium с картой-предписанием VRA
и растрами индексов (NDVI / NDRE / MSAVI / NDMI), динамика индексов по датам (Plotly),
баннер раннего стресса, скачивание GeoJSON и Shapefile для трактора.

## Запуск (Python 3.10+)

```bash
pip install -r requirements.txt
python generate_mock_data.py      # создаёт mock_data/ (6 недель, тестовое поле)
streamlit run app.py
```

## Источники данных

В боковой панели переключатель «Источник данных»:

- **Данные проекта (data/)** — папка `data/` репозитория; ищется автоматически вверх от `app.py`
  (до 4 уровней), поэтому модуль можно положить, например, в `modules/module4/`.
  Можно задать явно: `PAS_DATA_DIR=путь/к/data`.
- **Мок-данные** — `mock_data/` из `generate_mock_data.py`.

По умолчанию выбирается первый источник, где есть хотя бы одна дата.

## Что читается (всё необязательно — дашборд показывает то, что уже есть)

| Файл | Модуль | Использование |
|---|---|---|
| `field_boundary.geojson` или `samples/field_boundary.geojson` | M1 | контур поля, центрирование карты |
| `stats/index_stats_YYYY_MM_DD.json` (schema 1.1) | M2 | `mean_ndvi/ndre/msavi/ndmi` → график; `early_stress_alert`, `early_stress_area_ha/percent`, `early_stress.reasons` → баннер; `indices.*.p5/p95` → единая цветовая шкала на сезон; `zones.classes[].area_ha` → экономика, пока нет M3; `files.*` → пути к растрам |
| `indices/{NDVI,NDRE,MSAVI,NDMI}_YYYY_MM_DD.tif` | M2 | слои индексов (перепроецируются из UTM в WGS84) |
| `zones/zones_YYYY_MM_DD.tif` | M2 | растровая карта зон, пока нет полигонов M3 |
| `vra/` или `prescriptions/` или корень: `prescription_*.geojson`, `shapefile_*.zip`, `metrics_*.json` | M3 | полигоны с нормами, скачивание файлов, экономика |

Норма зоны в GeoJSON читается из `fertilizer_rate` (как в `data/README.md`) или `rate_kg_ha`.
Экономика пересчитывается под слайдеры: норма зоны = базовая норма × `rate_factor`
(если `rate_factor` нет в `metrics_*.json`, он считается как норма зоны / `base_rate_kg_ha`;
если нет и этого — 100% / 53% / 0% для зон 1 / 2 / 3).

## Вопросы к команде
- **M3:** в какую папку пишутся `prescription_*.geojson`, `shapefile_*.zip`, `metrics_*.json`? Дашборд ищет в `data/vra/`, `data/prescriptions/` и `data/`. Просьба добавить в `metrics_*.json` поля `base_rate_kg_ha`, `price_per_kg` и `zone_breakdown[].rate_factor`.
- **M2:** в каких единицах `early_stress.ndre_drop / ndvi_drop`? Сейчас дашборд их не показывает, выводит только `reasons` и площадь скрытого стресса.
