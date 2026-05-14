from utils.ollama_client import query_ollama
from rag.vector_store import add_documents
from utils.web_scraper import scrape_webpage


def extract_information(search_results, user_query):

    combined_text = ""
    seen_links = set()

    print(f"\n{'='*50}")
    print(f"Total search results received: {len(search_results)}")
    print(f"Query: {user_query}")
    print(f"{'='*50}\n")

    for r in search_results[:5]:

        link = r["link"]
        print(f"--- Processing: {link}")

        if link in seen_links:
            print("Skipped: duplicate link")
            continue

        seen_links.add(link)

        webpage_text = scrape_webpage(link)
        print(f"Scraped length: {len(webpage_text)} chars")

        if not webpage_text:
            snippet = r.get("snippet", "")
            print(f"Scrape empty, trying snippet: {len(snippet)} chars")
            webpage_text = snippet

        if not webpage_text:
            print("Both scrape and snippet empty, skipping")
            continue

        important_words = user_query.lower().split()
        relevance = any(
            word in webpage_text.lower()
            for word in important_words
        )
        print(f"Relevance check passed: {relevance}")

        if not relevance:
            print("Skipped: failed relevance filter")
            continue

        combined_text += f"""
        TITLE:
        {r['title']}

        CONTENT:
        {webpage_text}
        """

    print(f"\n{'='*50}")
    print(f"combined_text total length: {len(combined_text)}")
    print(f"Preview:\n{combined_text[:300]}")
    print(f"{'='*50}\n")

    if not combined_text.strip():
        return "Extraction failed: no content could be scraped from search results."

    prompt = f"""You are a research extraction agent.

Query: {user_query}

Source material:
{combined_text}

Extract ONLY concrete, verifiable facts from the sources above.
For each finding, include:
- The specific claim or data point
- Which source it comes from (use the TITLE)
- Why it is relevant to the query

Do NOT summarise. Do NOT use phrases like "the article discusses".
Pull out named figures, dates, product names, deal values, and direct statements."""

    extracted = query_ollama(prompt, model="llama3")

    add_documents([extracted])

    return extracted