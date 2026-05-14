import requests
from bs4 import BeautifulSoup


def scrape_webpage(url):

    try:

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        content_root = (
            soup.find("article")
            or soup.find("main")
            or soup.find("body")
            or soup
        )

        paragraphs = content_root.find_all("p")[:30]

        cleaned = []

        for p in paragraphs:

            text = p.get_text().strip()

            # Remove tiny/noisy paragraphs
            if len(text) > 50:

                cleaned.append(text)

        # 3 000 chars gives the extractor enough signal without
        # overflowing the Ollama context window.
        return "\n".join(cleaned)[:3000]

    except Exception:

        return ""