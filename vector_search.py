#!/usr/bin/env python3
"""
Vector Search Engine на основе TF-IDF и ручного вычисления косинусного сходства

Задача: используя ранее рассчитанные TF-IDF по термам и/или леммам,
построить векторный поиск без использования FAISS для метрики.

Структура папок:
- output/                # HTML-документы (для извлечения идентификаторов)
- tfidf_terms/           # Файлы <doc_id>_tfidf_terms.txt
- tfidf_lemmas/          # Файлы <doc_id>_tfidf_lemmas.txt

Результаты:
- tfidf_matrix.npy       # Документы × признаки (нормализованный TF-IDF)
- doc_ids.npy            # Список doc_id
- vocab.npy              # Словарь признаков

Использование:
    python vector_search_engine.py [--mode terms|lemmas] [--rebuild] [--top-k N]

Опции:
    --mode       : использовать 'terms' или 'lemmas' TF-IDF (по умолчанию 'terms')
    --rebuild    : пересоздать матрицу (по умолчанию загружает существующие)
    --top-k N    : число выдаваемых документов (по умолчанию 5)
"""
import argparse
import numpy as np
from pathlib import Path

# Константы папок и файлов
TFIDF_TERMS_DIR = 'tfidf_terms'
TFIDF_LEMMAS_DIR = 'tfidf_lemmas'
MAT_FILE = 'tfidf_matrix.npy'
DOCIDS_FILE = 'doc_ids.npy'
VOCAB_FILE = 'vocab.npy'


def load_tfidf(folder: Path):
    """Загрузить TF-IDF: возвращает doc_ids, vocab, tfidf_data"""
    tfidf_data = {}  # doc_id -> {term: tfidf}
    vocab = {}       # term -> idx
    doc_ids = []
    for path in sorted(folder.glob('*_tfidf_*.txt')):
        doc_id = path.stem.replace('_tfidf_terms','').replace('_tfidf_lemmas','')
        doc_ids.append(doc_id)
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                term, *_ = line.strip().split()
                tfidf_val = float(line.strip().split()[-1])
                if term not in vocab:
                    vocab[term] = len(vocab)
                tfidf_data.setdefault(doc_id, {})[term] = tfidf_val
    return doc_ids, list(vocab.keys()), tfidf_data


def build_matrix(doc_ids, vocab, tfidf_data):
    """Построить и нормализовать матрицу (D, V)"""
    D, V = len(doc_ids), len(vocab)
    mat = np.zeros((D, V), dtype=np.float32)
    term_to_idx = {t:i for i,t in enumerate(vocab)}
    for i, doc in enumerate(doc_ids):
        for term, val in tfidf_data.get(doc, {}).items():
            j = term_to_idx[term]
            mat[i, j] = val
    # L2-нормализация по строкам
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    return mat


def save_objects(mat, doc_ids, vocab):
    np.save(MAT_FILE, mat)
    np.save(DOCIDS_FILE, np.array(doc_ids))
    np.save(VOCAB_FILE, np.array(vocab))
    print(f"Сохранено: {MAT_FILE}, {DOCIDS_FILE}, {VOCAB_FILE}")


def load_objects():
    if not (Path(MAT_FILE).exists() and Path(DOCIDS_FILE).exists() and Path(VOCAB_FILE).exists()):
        return None, None, None
    mat = np.load(MAT_FILE)
    doc_ids = np.load(DOCIDS_FILE).tolist()
    vocab = np.load(VOCAB_FILE, allow_pickle=True).tolist()
    print("Загружены матрица TF-IDF, doc_ids и словарь vocab")
    return mat, doc_ids, vocab


def cosine_search(mat, query_vec, top_k):
    """Вычислить косинусное сходство вручную и вернуть индексы топ-K"""
    # query_vec уже нормирован
    sims = mat.dot(query_vec.flatten())  # shape (D,)
    # Получаем top_k индексы по убыванию
    idxs = np.argsort(-sims)[:top_k]
    return sims[idxs], idxs


def search_loop(mat, doc_ids, vocab, top_k):
    print("Введите запрос: список терминов через пробел:")
    term_to_idx = {t:i for i,t in enumerate(vocab)}
    while True:
        line = input('> ').strip()
        if not line:
            break
        q_terms = line.split()
        vec = np.zeros((mat.shape[1],), dtype=np.float32)
        count = 0
        for t in q_terms:
            idx = term_to_idx.get(t)
            if idx is not None:
                vec[idx] = 1.0
                count += 1
        if count == 0:
            print("Ни один термин не найден в словаре")
            continue
        vec /= count  # среднее
        # L2-нормализация запроса
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        sims, idxs = cosine_search(mat, vec.reshape(1, -1), top_k)
        print(f"Топ-{top_k} результатов:")
        for score, i in zip(sims, idxs):
            print(f"[{score:.4f}] {doc_ids[i]}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['terms','lemmas'], default='terms')
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()

    folder = Path(TFIDF_TERMS_DIR if args.mode=='terms' else TFIDF_LEMMAS_DIR)
    mat, doc_ids, vocab = (None, None, None) if args.rebuild else load_objects()

    if mat is None:
        doc_ids, vocab, tfidf_data = load_tfidf(folder)
        mat = build_matrix(doc_ids, vocab, tfidf_data)
        save_objects(mat, doc_ids, vocab)

    search_loop(mat, doc_ids, vocab, args.top_k)

if __name__ == '__main__':
    main()
