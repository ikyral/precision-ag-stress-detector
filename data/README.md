# Форматы данных

- M1 -> M2: `scene_YYYY_MM_DD.tif` (многоканальный, UTM)
- M2 -> M3: `zones_YYYY_MM_DD.tif` (uint8: 0 фон, 1 сильный стресс, 2 умеренный, 3 норма)
- M3 -> M4: `prescription_YYYY_MM_DD.geojson` (zone_class, area_ha, fertilizer_rate) и `metrics_YYYY_MM_DD.json`

## sample_scene.tif
- Каналы (10 шт.): 1:B02, 2:B03, 3:B04, 4:B05, 5:B06, 6:B07, 7:B08, 8:B8A, 9:B11, 10:B12
- Соответствие для spyndex: B, G, R, RE1, RE2, RE3, N, N2, S1, S2
- Тип: uint16, значения = отражение x 10000
- nodata = 0 (вне контура поля)
- Разрешение: 10 м
- CRS: UTM, зона EPSG:XXXXX (заполнить после скачивания снимка)

## sample_zones.tif
- Тип uint8: 0 вне поля, 1 сильный стресс, 2 умеренный, 3 норма
- Та же сетка и CRS, что у sample_scene.tif

## Эталоны
Лежат в `data/samples/`. Остальные данные в git не хранятся.
