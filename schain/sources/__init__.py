"""Retrieval providers. Each turns a query into candidate documents, nothing more."""

from . import gdelt, general_web, google_news, wikipedia

# Provider name in config -> search coroutine.
PROVIDERS = {
    "wikipedia": wikipedia.search,
    "gdelt": gdelt.search,
    "google_news": google_news.search,
    "general_web": general_web.search,
}

# Coarse label recorded in the retrieval manifest. Not a classifier: a fixed mapping.
SOURCE_TYPES = {
    "google_news": "NEWS",
    "gdelt": "NEWS",
    "general_web": "GENERAL_WEB",
    "wikipedia": "WIKIPEDIA",
}

__all__ = ["PROVIDERS", "SOURCE_TYPES", "gdelt", "general_web", "google_news", "wikipedia"]
