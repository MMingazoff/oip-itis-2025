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

## Задание 4
Для запуска

```commandline
pip install -r requirements.txt
python tf_idf.py
```

В папке `/tfidf_terms` находятся значения idf и tf-idf для терминов</br>
В папке `/tfidf_lemmas` находятся значения idf и tf-idf для лемм


## Задание 5

### Демонстрация векторного поиска

Кратко о запуске и использовании скрипта `vector_search.py`.

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

```bash
python vector_search_engine.py [--mode terms|lemmas] [--rebuild] [--top-k N]
```

Опции:</br>
    `--mode`      : использовать `terms` или `lemmas` TF-IDF (по умолчанию `terms`)</br>
    `--rebuild`    : пересоздать матрицу и индекс (по умолчанию загружает существующие)</br>
    `--top-k N`    : число выдаваемых документов (по умолчанию 5)
---

## Задание 6

### Поисковая система на основе TF-IDF и FAISS

Этот проект реализует векторную поисковую систему по HTML-документам с использованием TF-IDF-признаков и библиотеки FAISS. К системе подключён простой веб-интерфейс на Flask.


### Установка зависимостей

```bash
pip install -r requirements.txt
```
---

### Как запустить веб-интерфейс

1. Убедитесь, что файлы `tfidf_matrix.npy`, `doc_ids.npy`, `vocab.npy` и `vector.index` созданы с помощью:

```bash
python vector_search_engine.py --mode terms --rebuild
```

2. Запустите веб-приложение:

```bash
python web_search_app.py
```

3. Перейдите в браузере на [http://127.0.0.1:5000](http://127.0.0.1:5000) и введите поисковый запрос.

---

### Пример запроса

Введите список терминов через пробел:

```
интернет технологии история
```

---

### Примечания

* Для поиска по леммам используйте `--mode lemmas` при создании TF-IDF и индекса.
* Вектор запроса строится на основе совпадений с признаками из словаря `vocab.npy`.
