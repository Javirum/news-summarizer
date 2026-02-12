"""Response caching for news summarizer."""
import json
import os
from config import Config


class CacheStats:
    """Track cache hit/miss statistics."""

    def __init__(self):
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self):
        """Calculate cache hit rate as a percentage."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100


class ResponseCache:
    """File-based response cache keyed by article URL."""

    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or Config.CACHE_DIR
        self.cache_file = os.path.join(self.cache_dir, "response_cache.json")
        self.stats = CacheStats()
        self._cache = self._load()

    def _load(self):
        """Load cache from disk, returning empty dict on failure."""
        try:
            with open(self.cache_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _save(self):
        """Persist cache to disk."""
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(self._cache, f, indent=2)

    def get(self, url):
        """Get cached result for a URL. Returns None on miss."""
        if url in self._cache:
            self.stats.hits += 1
            return self._cache[url]
        self.stats.misses += 1
        return None

    def set(self, url, result):
        """Store a result and persist to disk."""
        self._cache[url] = result
        self._save()

    def clear(self):
        """Empty the cache and persist."""
        self._cache = {}
        self._save()
