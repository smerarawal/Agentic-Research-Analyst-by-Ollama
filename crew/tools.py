# crew/tools.py — replace entire file

from crewai.tools import BaseTool
from utils.web_scraper import scrape_webpage
from ddgs import DDGS
from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    query: str = Field(description="The search query string")


class ScrapInput(BaseModel):
    url: str = Field(description="The full URL to scrape")


class SearchWebTool(BaseTool):
    name: str = "Search Web"
    description: str = "Search the web for a given query and return results."
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:

        results = []

        trusted_domains = [
            "techcrunch.com", "theverge.com", "wired.com",
            "arstechnica.com", "venturebeat.com", "zdnet.com",
            "technologyreview.com", "axios.com", "fortune.com",
            "businessinsider.com"
        ]

        blocked_words = [
            "dictionary", "definition", "cambridge",
            "merriam", "wiktionary", "price in india"
        ]

        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5):
                    title   = r.get("title", "")
                    snippet = r.get("body", "")
                    link    = r.get("href", "")
                    combined = (title + snippet + link).lower()

                    if any(w in combined for w in blocked_words):
                        continue

                    score = 3 if any(d in link for d in trusted_domains) else 0
                    results.append((score, f"Title: {title}\nLink: {link}\n{snippet}"))

            results.sort(reverse=True)
            output = [r[1] for r in results[:5]]
            return "\n\n".join(output) if output else "No results found."

        except Exception as e:
            return f"Search failed: {e}"


class ScrapeWebpageTool(BaseTool):
    name: str = "Scrape Webpage"
    description: str = "Scrape and return the full text content of a webpage URL."
    args_schema: type[BaseModel] = ScrapInput

    def _run(self, url: str) -> str:
        try:
            return scrape_webpage(url) or "Could not scrape page."
        except Exception as e:
            return f"Scrape failed: {e}"


# Instantiate so agents.py can import them directly
search_tool  = SearchWebTool()
scrape_tool  = ScrapeWebpageTool()