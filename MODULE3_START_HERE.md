# Модуль 3 — Кузнецов Кирилл

Обновление от 28.10.2026 на базе архива main после PR #9 М2.
Подробный отчёт: [docs/module3_season_2026.md](docs/module3_season_2026.md).

## Запуск из корня проекта

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r src/module2/requirements.txt -r src/module3/requirements.txt -r src/module4/requirements.txt
.\.venv\Scripts\python.exe src/module3/examples/run_real_season.py
.\.venv\Scripts\python.exe src/module3/examples/verify_real_season.py
.\.venv\Scripts\python.exe -m streamlit run src/module4/app.py
```

Проверено на Python 3.12.14. Скрипт заново обрабатывает восемь исходных сцен
через М2 и М3. Результаты: `data/prescriptions/` (GeoJSON, Shapefile ZIP,
JSON-экономика, сводка и аудит покрытия). М2 также создаёт `data/stats`,
`data/indices`, `data/zones` для кабинета. Эти данные исключены из Git;
в полном архиве поставки они приложены, после клонирования их надо создать.

## Если М2 уже сформировал маски

```powershell
python -m pip install -e src/module3
python -m vra run data/zones -o data/prescriptions --reference-mask data/samples/sample_zones.tif
```

Нормы по умолчанию: 150/80/0 кг/га, база сравнения 150, цена 45 руб./кг.
Новый сценарий задаётся явно: `--rates 1=200,2=100,3=0 --base-rate 200 --price 50`.
Изменение ползунка в кабинете моделирует экономику, но не генерирует новые
файлы: при изменении норм скачивание старого задания отключено.

## Проверки

```powershell
python -m unittest discover -s src/module3/tests -v
python src/module3/examples/verify_real_season.py
```

М3 использует классификацию М2. Облачные участки, которым М2 присвоил класс
ближайшей зоны, включаются в предписание. На 18 июля восстановлено 4,12 га;
это явно отмечено в аудите и кабинете. На всех восьми датах площадь 79,6 га.
Оставшийся nodata М3 не заполняет. Нормы учебные; импорт в конкретный бортовой
компьютер и общий main_pipeline.py не входят в эти проверки.
