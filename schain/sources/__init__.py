"""Retrieval providers. Each turns a query into candidate documents, nothing more."""

from . import gdelt, google_news, wikipedia

# Provider name in config -> search coroutine.
PROVIDERS = {
    "wikipedia": wikipedia.search,
    "gdelt": gdelt.search,
    "google_news": google_news.search,
}

__all__ = ["PROVIDERS", "gdelt", "google_news", "wikipedia"]
