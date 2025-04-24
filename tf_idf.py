import os
import math
import re
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

output_folder = "output/"
lemmas_folder = "lemmas_per_doc/"
tf_idf_terms_folder = "tfidf_terms/"
tf_idf_lemmas_folder = "tfidf_lemmas/"

russian_stopwords = set(stopwords.words("russian"))

def ensure_directories():
    os.makedirs(tf_idf_terms_folder, exist_ok=True)
    os.makedirs(tf_idf_lemmas_folder, exist_ok=True)

def extract_text_from_html(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        return soup.get_text()

def tokenize(text):
    words = word_tokenize(text.lower())
    tokens = list()
    for word in words:
        if (word.isalpha() and # Только буквы
            word not in russian_stopwords and
            not re.search(r'\d', word) and # Без цифр
            re.fullmatch(r'^[А-Яа-яЁё]+$', word)): # Только русские слова
            tokens.append(word)
    return tokens

def load_lemmas():
    lemma_forms = {}
    lemma_df = defaultdict(int)
    for filename in os.listdir(lemmas_folder):
        if filename.endswith("_lemmas.txt"):
            with open(os.path.join(lemmas_folder, filename), "r", encoding="utf-8") as f:
                lemma_map = {}
                for line in f:
                    parts = line.strip().split(":")
                    if len(parts) == 2:
                        lemma, forms = parts
                        forms_list = forms.strip().split()
                        lemma_map[lemma] = forms_list
                lemma_forms[filename.replace("_lemmas.txt", "")] = lemma_map
                for lemma in lemma_map:
                    lemma_df[lemma] += 1
    return lemma_forms, lemma_df

def compute_tf_idf():
    ensure_directories()

    token_docs = {}
    token_df = defaultdict(int)
    filenames = [f for f in os.listdir(output_folder) if f.endswith(".html")]

    # TF and DF collection for terms
    for filename in filenames:
        full_path = os.path.join(output_folder, filename)
        text = extract_text_from_html(full_path)
        tokens = tokenize(text)
        token_docs[filename] = tokens
        for token in set(tokens):
            token_df[token] += 1

    N = len(token_docs)

    # Calculate TF-IDF for terms
    for filename, tokens in token_docs.items():
        term_counts = Counter(tokens)
        total_terms = sum(term_counts.values())
        output_path = os.path.join(tf_idf_terms_folder, filename.replace(".html", "_tfidf_terms.txt"))
        with open(output_path, "w", encoding="utf-8") as out:
            for term, count in term_counts.items():
                tf = count / total_terms
                idf = math.log((1 + N) / (1 + token_df[term]))
                tfidf = tf * idf
                out.write(f"{term} {idf:.6f} {tfidf:.6f}\n")

    # Load lemmas
    lemma_forms, lemma_df = load_lemmas()

    # Calculate TF-IDF for lemmas
    for filename in filenames:
        doc_id = filename.replace(".html", "")
        if doc_id not in lemma_forms:
            continue

        tokens = token_docs[filename]
        term_counts = Counter(tokens)
        total_terms = sum(term_counts.values())

        output_path = os.path.join(tf_idf_lemmas_folder, f"{doc_id}_tfidf_lemmas.txt")
        with open(output_path, "w", encoding="utf-8") as out:
            for lemma, forms in lemma_forms[doc_id].items():
                total_lemma_count = sum(term_counts.get(form, 0) for form in forms)
                if total_lemma_count == 0:
                    continue
                tf = total_lemma_count / total_terms
                idf = math.log((1 + N) / (1 + lemma_df[lemma]))
                tfidf = tf * idf
                out.write(f"{lemma} {idf:.6f} {tfidf:.6f}\n")

def main():
    compute_tf_idf()
    print("TF-IDF по HTML-документам успешно рассчитан.")

if __name__ == "__main__":
    main()
