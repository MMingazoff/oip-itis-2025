import os
import requests

def fetch_and_save(url, index):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        page_id = url.split("//")[-1].replace("/", "_").replace(":", "_").replace("?", "_").replace("=", "_")
        file_name = f"output/{index}_{page_id}.html"

        os.makedirs("output", exist_ok=True)

        with open(file_name, "w", encoding="utf-8") as f:
            f.write(response.text)

        with open("output/index.txt", "a", encoding="utf-8") as index_file:
            index_file.write(f"{file_name} {url}\n")

        print(f"[+] Saved: {url}")
    except Exception as e:
        print(f"[-] Failed: {url}, Error: {e}")


def main():
    with open("urls.txt", "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f.readlines()]

    for index, url in enumerate(urls):
        fetch_and_save(url, index)


if __name__ == "__main__":
    main()
