import os
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy2
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

morph = pymorphy2.MorphAnalyzer()
russian_stopwords = set(stopwords.words("russian"))

def clean_text_from_html(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
        return soup.get_text()

def tokenize(text):
    words = word_tokenize(text.lower())
    tokens = set()
    for word in words:
        if (word.isalpha() and # Только буквы
            word not in russian_stopwords and
            not re.search(r'\d', word) and # Без цифр
            re.fullmatch(r'^[А-Яа-яЁё]+$', word)): # Только русские слова
            tokens.add(word)
    return tokens

def lemmatize_tokens(tokens):
    lemmas = {}
    for token in tokens:
        lemma = morph.parse(token)[0].normal_form
        if lemma not in lemmas:
            lemmas[lemma] = []
        lemmas[lemma].append(token)
    return lemmas

def main():
    input_folder = "output/"
    tokens_folder = "tokens_per_doc/"
    lemmas_folder = "lemmas_per_doc/"

    os.makedirs(tokens_folder, exist_ok=True)
    os.makedirs(lemmas_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".html"):
            filepath = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]

            text = clean_text_from_html(filepath)
            tokens = tokenize(text)

            with open(os.path.join(tokens_folder, f"{base_name}_tokens.txt"), "w", encoding="utf-8") as f:
                for token in sorted(tokens):
                    f.write(token + "\n")

            lemmas = lemmatize_tokens(tokens)
            with open(os.path.join(lemmas_folder, f"{base_name}_lemmas.txt"), "w", encoding="utf-8") as f:
                for lemma, forms in lemmas.items():
                    f.write(f"{lemma}: {' '.join(sorted(set(forms)))}\n")

    print("Файлы токенов и лемм сохранены по документам.")

if __name__ == "__main__":
    main()
