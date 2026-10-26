# Модуль 3 — Кузнецов Кирилл

В этот проект добавлен готовый модуль генерации VRA-карт и экономического отчёта.
Основной код находится в `src/module3/vra/`, подробная инструкция — в
[`src/module3/README.md`](src/module3/README.md).

## Быстрый запуск в Windows

Откройте PowerShell в корне распакованного проекта:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r src/module3/requirements.txt
.\.venv\Scripts\python.exe src/module3/examples/run_samples.py
```

Нужен Python 3.11 или новее. Если команда `python` не находится, укажите полный
путь к установленному Python. На компьютере, использованном для проверки, это:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe" -m venv .venv
```

Далее используйте команды через `.\.venv\Scripts\python.exe`, как выше.
Это отдельное окружение проекта; установка библиотек не меняет системные пакеты.

## Готовые результаты

- `data/module3_demo/sample/` — карта по тестовой маске `sample_zones.tif`.
- `data/module3_demo/module2/` — карта по маске аналитического модуля `sample_zones_m2.tif`.
- `data/module3_demo/pipeline_m2_m3/` — результат полного прогона исходного снимка через модули 2 и 3.
- `data/module3_demo/synthetic_season/` — шесть искусственных недель для проверки пакетной обработки.
- В каждой папке результата есть GeoJSON, JSON с экономикой, ZIP и распакованные компоненты Shapefile.
- `validation/` — протоколы проверок поставки.

Второй пример подходит для демонстрации стыковки с модулем 2. Первый — тестовая
разбивка по NDVI. Это не подтверждённые очаги болезни. Нормы 150 / 80 / 0 кг/га и
цена 45 руб./кг взяты из учебного ТЗ.

## Запуск на новых масках

```powershell
.\.venv\Scripts\python.exe -m pip install -e src/module3
.\.venv\Scripts\python.exe -m vra run data/zones -o data/prescriptions --rates 1=150,2=80,3=0 --base-rate 150 --price 45
```

Каталог `data/zones` должен содержать `zones_YYYY_MM_DD.tif` для одного поля.
Для одиночного файла без даты в имени добавьте `--date 2026-08-07`.
Полный формат JSON, перерасчёт экономики, режим литров и настройки фильтрации
описаны в инструкции модуля.

## Проверки

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s src/module3/tests -v
.\.venv\Scripts\python.exe src/module3/examples/create_demo_season.py
```

В исходном архиве нет реального ряда за сезон и готового модуля 4. Генерация файлов
для них реализована; проверка отображения в будущем кабинете и импорта в конкретную
модель бортового компьютера остаётся этапом командной интеграции.
