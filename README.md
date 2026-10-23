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
Скопируйте `.env.example` в `.env` и впишите ключи Copernicus S3.

## Лицензия и данные
Код: MIT. Contains modified Copernicus Sentinel data 2026.
