import os
import re
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.fia.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}
KEYWORDS = ["decision", "infringement", "offence"]

SEASONS = {
    "2025": "https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-2025-2071",
    "2024": "https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-2024-2043",
    "2023": "https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-2023-2042",
    "2022": "https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-2022-2005",
    "2021": "https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-2021-1108",
    "2020": "https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-2020-1059",
    "2019": "https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-2019-971",
    "2015": "https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-2015-249"
}


def get_event_node_ids(season_url):
    response = requests.get(season_url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    events = {}
    for a in soup.select("a[href*='/decision-document-list/nojs/']"):
        match = re.search(r'/decision-document-list/nojs/(\d+)', a["href"])
        if match:
            node_id = match.group(1)
            name = a.get_text(strip=True)
            events[node_id] = name
    return events


def get_pdfs_for_event(node_id):
    url = f"{BASE_URL}/decision-document-list/ajax/{node_id}"
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    html_data = ""
    event_name = f"event_{node_id}"

    for command in response.json():
        if command.get("command") == "insert":
            selector = command.get("selector", "")
            data = command.get("data", "")
            if "event-title" in selector:
                text = BeautifulSoup(data, "html.parser").get_text(strip=True)
                if text:
                    event_name = text
            if "document-type-wrapper" in selector:
                html_data += data

    soup = BeautifulSoup(html_data, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if "/system/files/decision-document/" in href or "/sites/default/files/decision-document/" in href:
            title = a.get_text(" ", strip=True)
            links.append((href, title))

    return event_name, links


def scrape(season_url, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    events = get_event_node_ids(season_url)
    print(f"Found {len(events)} events: {list(events.values())}")

    for node_id, fallback_name in events.items():
        try:
            event_name, links = get_pdfs_for_event(node_id)
        except Exception as e:
            print(f"Failed node {node_id} ({fallback_name}): {e}")
            continue

        print(f"\n[{event_name}] {len(links)} PDFs found")

        event_folder = os.path.join(output_dir, event_name.replace("/", "-").replace(" ", "_"))
        os.makedirs(event_folder, exist_ok=True)

        downloaded = 0
        for href, title in links:
            if not any(k in title.lower() for k in KEYWORDS):
                continue

            pdf_url = BASE_URL + href if href.startswith("/") else href
            filename = href.split("/")[-1]
            filepath = os.path.join(event_folder, filename)

            if os.path.exists(filepath):
                continue

            try:
                r = requests.get(pdf_url, headers=HEADERS, timeout=20)
                r.raise_for_status()
                with open(filepath, "wb") as f:
                    f.write(r.content)
                print(f"  + {filename}")
                downloaded += 1
            except Exception as e:
                print(f"  Failed: {pdf_url} -> {e}")

        print(f"  Downloaded: {downloaded}")


if __name__ == "__main__":
    for year, url in SEASONS.items():
        print(f"\n{'='*50}")
        print(f"SEASON {year}")
        print(f"{'='*50}")
        scrape(url, f"./fia_pdfs/{year}")

    print("\nAll seasons complete.")