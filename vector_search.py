"""
Структура папок:
- output/                # HTML-документы (для извлечения идентификаторов)
- tfidf_terms/           # Файлы <doc_id>_tfidf_terms.txt
- tfidf_lemmas/          # Файлы <doc_id>_tfidf_lemmas.txt

Результаты:
- tfidf_matrix.npy       # Документы × признаки (TF-IDF)
- doc_ids.npy            # Список doc_id
- vector.index           # FAISS индекс

Использование:
    python vector_search_engine.py [--mode terms|lemmas] [--rebuild] [--top-k N]

Опции:
    --mode       : использовать 'terms' или 'lemmas' TF-IDF (по умолчанию 'terms')
    --rebuild    : пересоздать матрицу и индекс (по умолчанию загружает существующие)
    --top-k N    : число выдаваемых документов (по умолчанию 5)
"""
import os
import argparse
import numpy as np
from pathlib import Path
import faiss

# Константы папок и файлов
OUTPUT_DIR = 'output'
TFIDF_TERMS_DIR = 'tfidf_terms'
TFIDF_LEMMAS_DIR = 'tfidf_lemmas'
MAT_FILE = 'tfidf_matrix.npy'
DOCIDS_FILE = 'doc_ids.npy'
INDEX_FILE = 'vector.index'


def load_tfidf(folder: Path) -> (list, list, dict):
    """Загрузить TF-IDF из файлов в папке: возвращает doc_ids, vocab, tfidf_data"""
    tfidf_data = {}  # doc_id -> {term: tfidf}
    vocab = {}       # term -> idx
    doc_ids = []
    for path in sorted(folder.glob('*_tfidf_*.txt')):
        doc_id = path.stem.replace('_tfidf_terms','').replace('_tfidf_lemmas','')
        doc_ids.append(doc_id)
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                term, idf, tfidf = line.strip().split()
                tfidf_val = float(tfidf)
                if term not in vocab:
                    vocab[term] = len(vocab)
                tfidf_data.setdefault(doc_id, {})[term] = tfidf_val
    return doc_ids, list(vocab.keys()), tfidf_data


def build_matrix(doc_ids: list, vocab: list, tfidf_data: dict) -> np.ndarray:
    """Построить матрицу размером (len(doc_ids), len(vocab))"""
    D, V = len(doc_ids), len(vocab)
    mat = np.zeros((D, V), dtype=np.float32)
    term_to_idx = {t:i for i,t in enumerate(vocab)}
    for i, doc in enumerate(doc_ids):
        for term, val in tfidf_data.get(doc, {}).items():
            j = term_to_idx[term]
            mat[i, j] = val
    return mat


def save_index_objects(mat: np.ndarray, doc_ids: list, index):
    np.save(MAT_FILE, mat)
    np.save(DOCIDS_FILE, np.array(doc_ids))
    faiss.write_index(index, INDEX_FILE)
    print(f"Сохранено: {MAT_FILE}, {DOCIDS_FILE}, {INDEX_FILE}")


def load_index_objects():
    if not (Path(MAT_FILE).exists() and Path(DOCIDS_FILE).exists() and Path(INDEX_FILE).exists()):
        return None, None, None
    mat = np.load(MAT_FILE)
    doc_ids = np.load(DOCIDS_FILE).tolist()
    index = faiss.read_index(INDEX_FILE)
    print("Загружены матрица TF-IDF и индекс")
    return mat, doc_ids, index


def build_faiss(mat: np.ndarray) -> faiss.IndexFlatIP:
    """Нормализация и создание FAISS IndexFlatIP"""
    faiss.normalize_L2(mat)
    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(mat)
    return index


def search_loop(index, doc_ids, top_k: int):
    print("Введите запрос в формате списка терминов через пробел (или стоп-словосочетание):")
    while True:
        line = input('> ').strip()
        if not line:
            break
        # запрос разбивается на термы: конструируем вектор совпадений
        q_terms = line.split()
        # пост-фильтрация: усредняем единичные векторы
        # пропускаем термы вне словаря
        # строим вектор запросов
        vec = np.zeros(index.d, dtype=np.float32)
        # assume same vocab as loaded matrix
        # load vocab from file
        vocab_keys = np.load('vocab.npy', allow_pickle=True).tolist()
        term_to_idx = {t:i for i,t in enumerate(vocab_keys)}
        count = 0
        for t in q_terms:
            idx = term_to_idx.get(t)
            if idx is not None:
                vec[idx] = 1.0
                count += 1
        if count > 0:
            vec /= count
            # normalize
            vec = vec.reshape(1, -1)
            faiss.normalize_L2(vec)
            D, I = index.search(vec, top_k)
            print(f"Топ-{top_k} результатов:")
            for score, idx in zip(D[0], I[0]):
                print(f"[{score:.4f}] {doc_ids[idx]}")
        else:
            print("Ни один термин запроса не найден в словаре")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['terms','lemmas'], default='terms', help='Использовать TF-IDF по термам или леммам')
    parser.add_argument('--rebuild', action='store_true', help='Пересоздать матрицу и индекс')
    parser.add_argument('--top-k', type=int, default=5, help='Число выдаваемых документов')
    args = parser.parse_args()

    folder = Path(TFIDF_TERMS_DIR if args.mode=='terms' else TFIDF_LEMMAS_DIR)
    mat, doc_ids, index = (None, None, None) if args.rebuild else load_index_objects()

    if mat is None or args.rebuild:
        doc_ids, vocab, tfidf_data = load_tfidf(folder)
        # сохраняем словарь для запросов
        np.save('vocab.npy', np.array(vocab))
        mat = build_matrix(doc_ids, vocab, tfidf_data)
        index = build_faiss(mat)
        save_index_objects(mat, doc_ids, index)

    search_loop(index, doc_ids, args.top_k)

if __name__ == '__main__':
    main()
