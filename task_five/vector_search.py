import os
import sys
import argparse
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

INDEX_PATH = 'vector.index'
DOCS_DIR = '/documents'

def load_documents(path):
    docs = []
    names = []
    for fname in os.listdir(path):
        full = os.path.join(path, fname)
        if os.path.isfile(full) and fname.lower().endswith('.txt'):
            with open(full, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            if text:
                docs.append(text)
                names.append(fname)
    return docs, names

def build_index(embeddings: np.ndarray, dim: int):
    # используем косинусное сходство: normalize + IndexFlatIP
    faiss.normalize_L2(embeddings)
    idx = faiss.IndexFlatIP(dim)
    idx.add(embeddings)
    return idx

def save_index(idx, path):
    faiss.write_index(idx, path)
    print(f"Индекс сохранён в {path}")

def load_index(path):
    if not os.path.exists(path):
        return None
    idx = faiss.read_index(path)
    print(f"Индекс загружен из {path}")
    return idx

def main(args):
    # 1) загружаем или строим индекс
    model = SentenceTransformer('all-MiniLM-L6-v2')
    idx = load_index(INDEX_PATH)

    if idx is None:
        print("Строим индекс заново…")
        docs, names = load_documents(DOCS_DIR)
        if not docs:
            print(f"Нет файлов .txt в папке {DOCS_DIR}")
            sys.exit(1)
        print(f"Загружено документов: {len(docs)}")
        embs = model.encode(docs, show_progress_bar=True, normalize_embeddings=True)
        idx = build_index(embs, embs.shape[1])
        save_index(idx, INDEX_PATH)
        # сохраняем mapping
        np.save('names.npy', np.array(names))
    else:
        names = list(np.load('names.npy'))

    # 2) поиск в интерактивном режиме
    print("\nВведите текст запроса (или пустую строку для выхода):")
    while True:
        q = input("> ").strip()
        if not q:
            break
        q_emb = model.encode([q], normalize_embeddings=True)
        top_k = args.top_k
        D, I = idx.search(q_emb, top_k)
        print(f"\nТоп-{top_k} результатов:")
        for score, idx_doc in zip(D[0], I[0]):
            print(f"  [{score:.4f}] {names[idx_doc]}")
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vector Search Demo")
    parser.add_argument('--top-k', type=int, default=5, help='Число документов в выдаче')
    args = parser.parse_args()
    main(args)
