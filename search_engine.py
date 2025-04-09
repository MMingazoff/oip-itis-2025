import re
from collections import defaultdict

class BooleanSearchEngine:
    """
    Класс для реализации булева поиска по инвертированному индексу.
    Поддерживает операторы AND, OR, NOT и сложные запросы со скобками.
    """
    def __init__(self):
        # Инвертированный индекс: слово -> множество документов
        self.index = defaultdict(set)
        # Хранилище документов: doc_id -> текст документа
        self.documents = {}
    
    def add_document(self, doc_id, text):
        """
        Добавляет документ в индекс.
        :param doc_id: уникальный идентификатор документа
        :param text: текст документа для индексации
        """
        self.documents[doc_id] = text
        # Разбиваем текст на слова и добавляем в индекс
        for word in self._tokenize(text):
            self.index[word].add(doc_id)
    
    def _tokenize(self, text):
        """
        Токенизация текста - разбиение на слова.
        :param text: входной текст
        :return: список слов в нижнем регистре
        """
        return re.findall(r'\w+', text.lower())
    
    def search(self, query):
        """
        Выполняет булев поиск по запросу.
        :param query: поисковый запрос с операторами
        :return: множество doc_id соответствующих документов
        """
        # Обработка вложенных скобок рекурсивно
        while '(' in query:
            query = re.sub(r'\(([^()]+)\)', lambda m: self._process_group(m.group(1)), query)
        
        # Разбиваем запрос на токены (операторы и слова)
        tokens = re.findall(r'AND|OR|NOT|\w+', query, re.IGNORECASE)
        
        if not tokens:
            return set()
        
        # Обработка NOT в начале запроса
        if tokens[0].upper() == 'NOT':
            if len(tokens) < 2:
                return set()
            return self._not_operation(self._get_docs(tokens[1]))
        
        # Начинаем с первого термина
        result = self._get_docs(tokens[0])
        
        # Обрабатываем последующие операторы и термины
        for i in range(1, len(tokens), 2):
            if i+1 >= len(tokens):
                break
                
            operator = tokens[i].upper()
            term = tokens[i+1]
            docs = self._get_docs(term)
            
            if operator == 'AND':
                result = self._and_operation(result, docs)
            elif operator == 'OR':
                result = self._or_operation(result, docs)
            elif operator == 'NOT':
                result = self._and_operation(result, self._not_operation(docs))
        
        return result
    
    def _process_group(self, group_query):
        """
        Обрабатывает подзапрос в скобках.
        :param group_query: подзапрос внутри скобок
        :return: временный ключ для замены группы
        """
        group_result = self.search(group_query)
        key = f"GROUP_{abs(hash(group_query))}"
        self.index[key] = group_result
        return key
    
    def _get_docs(self, term):
        """
        Возвращает документы для термина или группы.
        :param term: слово или ключ группы
        :return: множество doc_id
        """
        if term.startswith('GROUP_'):
            return self.index.get(term, set())
        return self.index.get(term.lower(), set())
    
    def _and_operation(self, set1, set2):
        """Логическое И (пересечение множеств)"""
        return set1 & set2
    
    def _or_operation(self, set1, set2):
        """Логическое ИЛИ (объединение множеств)"""
        return set1 | set2
    
    def _not_operation(self, docs):
        """Логическое НЕ (дополнение множества)"""
        all_docs = set(self.documents.keys())
        return all_docs - docs
    
    def save_index_to_file(self, filename):
        """
        Сохраняет инвертированный индекс в файл.
        :param filename: имя файла для сохранения
        """
        with open(filename, 'w', encoding='utf-8') as f:
            for term in sorted(self.index.keys()):
                docs = ','.join(sorted(self.index[term]))
                f.write(f"{term}:{docs}\n")

def load_lemmas(file_path):
    """
    Загружает леммы из файла.
    :param file_path: путь к файлу с леммами
    :return: словарь {лемма: формы слова}
    """
    lemmas = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                lemma, forms = line.strip().split(':', 1)
                lemmas[lemma] = forms.strip()
    return lemmas

def build_index(lemma_files):
    """
    Строит инвертированный индекс из файлов с леммами.
    :param lemma_files: список путей к файлам
    :return: объект BooleanSearchEngine с построенным индексом
    """
    engine = BooleanSearchEngine()
    for file_num, file_path in enumerate(lemma_files, 1):
        lemmas = load_lemmas(file_path)
        for lemma, forms in lemmas.items():
            doc_id = f"{file_num}_{lemma}"
            engine.add_document(doc_id, f"{lemma} {forms}")
    return engine

def demonstrate_search(engine):
    """
    Демонстрация работы булева поиска.
    :param engine: объект BooleanSearchEngine
    """
    # Пример документов в индексе (для наглядности)
    sample_docs = {
        "1_разработчик": "разработчик разработчиками разработчик разработчика",
        "2_сервис": "сервис сервисами сервисом сервисов",
        "1_технология": "технология технологиям технологий",
        "2_инновация": "инновация инновациям"
    }
    
    # Тестовые запросы и ожидаемые результаты
    test_cases = [
        ("разработчик AND сервис", {"1_разработчик", "2_сервис"}),
        ("технология OR инновация", {"1_технология", "2_инновация"}),
        ("разработчик NOT сервис", {"1_разработчик"}),
        ("(технология AND инновация) OR разработчик", {"1_технология", "2_инновация", "1_разработчик"})
    ]
    
    print("Демонстрация булева поиска:")
    for query, expected in test_cases:
        print(f"\nЗапрос: '{query}'")
        print("Ожидаемый результат:", sorted(expected))
        
        result = engine.search(query)
        print("Фактический результат:", sorted(result))
        
        if result == expected:
            print("Тест пройден")
        else:
            print("Тест не пройден")

if __name__ == "__main__":
    # 1. Строим индекс из файлов
    engine = build_index(['lemmas_1.txt', 'lemmas_2.txt'])
    
    # 2. Сохраняем инвертированный индекс
    engine.save_index_to_file('inverted_index.txt')
    print("Инвертированный индекс сохранен в inverted_index.txt")
    
    # 3. Демонстрация работы поиска
    demonstrate_search(engine)