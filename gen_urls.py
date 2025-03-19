import requests


def get_random_wikipedia_urls(count=100):
    url = "https://ru.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": count
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    base_url = "https://ru.wikipedia.org/wiki/"
    return [base_url + page["title"].replace(" ", "_") for page in data["query"]["random"]]


def save_urls_to_file(urls, filename="urls.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for url in urls:
            f.write(url + "\n")


def main():
    urls = get_random_wikipedia_urls()
    save_urls_to_file(urls)
    print(f"Сгенерировано {len(urls)} уникальных ссылок на страницы Википедии и сохранено в urls.txt")


if __name__ == "__main__":
    main()