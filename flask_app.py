# web_search_app.py
from flask import Flask, render_template, request
import numpy as np
import faiss
from pathlib import Path

# --- Загружаем объекты поиска ---
MAT_FILE = 'tfidf_matrix.npy'
DOCIDS_FILE = 'doc_ids.npy'
INDEX_FILE = 'vector.index'
VOCAB_FILE = 'vocab.npy'

mat = np.load(MAT_FILE)
doc_ids = np.load(DOCIDS_FILE).tolist()
index = faiss.read_index(INDEX_FILE)
vocab = np.load(VOCAB_FILE, allow_pickle=True).tolist()
term_to_idx = {t: i for i, t in enumerate(vocab)}

# --- Flask ---
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def search():
    results = []
    query = ''
    if request.method == 'POST':
        query = request.form['query']
        terms = query.strip().split()
        vec = np.zeros(index.d, dtype=np.float32)
        count = 0
        for t in terms:
            idx = term_to_idx.get(t)
            if idx is not None:
                vec[idx] = 1.0
                count += 1
        if count > 0:
            vec /= count
            vec = vec.reshape(1, -1)
            faiss.normalize_L2(vec)
            D, I = index.search(vec, 10)
            results = [(doc_ids[i], float(score)) for score, i in zip(D[0], I[0])]

    return render_template('index.html', results=results, query=query)

if __name__ == '__main__':
    app.run(debug=True)
