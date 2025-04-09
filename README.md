# ОИП 2025

Студенты: 
- Мингазов Марат Фанисович 11-104
- Хамидуллов Ринат Ришатович 11-104

## Задание 1
Для запуска

```commandline
pip install -r requirements.txt
python downloader.py
```

Вывод в папке `/outputs`, там же и находится `index.txt` файл

Также в `gen_url.py` хранится код для генерации 100 случайных страниц в википедии. 
Его запускать не нужно, список ссылок уже сгенерирован

## Задание 2
Для запуска

```commandline
pip install -r requirements.txt
python tokens_lemmas.py
```

Токены для каждого документа из `/outputs` находятся в папке `/tokens_per_doc`<br/>
Леммы для каждого документа из `/outputs` находятся в папке `/lemmas_per_doc`. Строятся на основе токенов.

## Задание 3
Для запуска

```commandline
pip install -r requirements.txt
python search_engine.py
```

Леммы взяты со второго задания (которые были прикреплены к заданию на edu). Для запуска поместить в одну директорию с search_engine.py
Файл с инвертированным индексом inverted_index.txt
