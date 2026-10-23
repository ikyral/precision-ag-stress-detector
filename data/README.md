# Форматы данных

- M1 -> M2: `scene_YYYY_MM_DD.tif` (многоканальный, UTM или EPSG:4326)
- M2 -> M3: `zones_YYYY_MM_DD.tif` (uint8: 0 фон, 1 сильный стресс, 2 умеренный, 3 норма)
- M3 -> M4: `prescription_YYYY_MM_DD.geojson` (zone_class, area_ha, fertilizer_rate) и `metrics_YYYY_MM_DD.json`

## Порядок каналов sample_scene.tif
1:B02, 2:B03, 3:B04, 4:B05, 5:B06, 6:B07, 7:B08, 8:B8A, 9:B11, 10:B12
(заполнить: CRS, масштаб значений, nodata)

## Эталоны
Лежат в `data/samples/`. Остальные данные в git не хранятся.
