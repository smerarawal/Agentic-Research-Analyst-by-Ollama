from ddgs import DDGS

def search_web(query, max_results=5):

    results_list = []

    blocked_words = [
        "dictionary",
        "definition",
        "cambridge",
        "merriam",
        "researchgate",
        "wiktionary",
        "price in india",
        "dealers"
    ]

    trusted_domains = [
        "techcrunch.com",
        "theverge.com",
        "wired.com",
        "arstechnica.com",
        "venturebeat.com",
        "zdnet.com",
        "technologyreview.com",
        "axios.com",
        "fortune.com",
        "businessinsider.com"
    ]

    # Use the planner's task query directly — it's already specific.
    # Adding a generic suffix forces DuckDuckGo into broad results
    # regardless of what the planner asked for.
    with DDGS() as ddgs:

        results = ddgs.text(
            query,
            max_results=max_results
        )

        for r in results:

            title = r.get("title", "")
            snippet = r.get("body", "")
            link = r.get("href", "")

            combined = (
                title + " " + snippet + " " + link
            ).lower()

            # Remove junk pages
            if any(
                word in combined
                for word in blocked_words
            ):
                continue

            score = 0

            # Trusted domain scoring
            if any(
                domain in link
                for domain in trusted_domains
            ):
                score += 3

            # Generic business/research relevance
            important_keywords = [
                "strategy",
                "technology",
                "market",
                "competition",
                "financial",
                "ai",
                "products",
                "partnerships",
                "regulation",
                "infrastructure"
            ]

            if any(
                word in combined
                for word in important_keywords
            ):
                score += 2

            if "wikipedia" in link:
                score -= 1

            results_list.append({
                "title": title,
                "snippet": snippet,
                "link": link,
                "score": score
            })

    results_list.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results_list[:5]