
import logging

from ddgs import DDGS

log = logging.getLogger("web_search")


def search_context(query: str, max_results: int = 6) -> str:
    """Returns short plain-text results, or "" if search failed or found nothing."""
    try:
        # Use DDGS as a context manager for efficient session management and connection cleanup
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results, timeout=5)
    except Exception as e:
        log.warning("DuckDuckGo search failed: %s", e)
        return ""
        
    if not results:
        return ""
        
    return "\n".join(f"- {r.get('title', '')}: {r.get('body', '')}" for r in results if r)