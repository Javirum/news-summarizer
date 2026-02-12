"""Unit tests for news summarizer."""
import os
import pytest
from unittest.mock import Mock, patch
from news_api import NewsAPI
from llm_providers import LLMProviders, CostTracker, count_tokens
from summarizer import NewsSummarizer
from cache import CacheStats, ResponseCache


class TestCostTracker:
    """Test cost tracking functionality."""

    def test_track_request(self):
        """Test tracking a single request."""
        tracker = CostTracker()
        cost = tracker.track_request("openai", "gpt-4o-mini", 100, 500)

        assert cost > 0
        assert tracker.total_cost == cost
        assert len(tracker.requests) == 1

    def test_get_summary(self):
        """Test summary generation."""
        tracker = CostTracker()
        tracker.track_request("openai", "gpt-4o-mini", 100, 200)
        tracker.track_request("anthropic", "claude-3-5-sonnet-20241022", 150, 300)

        summary = tracker.get_summary()

        assert summary["total_requests"] == 2
        assert summary["total_cost"] > 0
        assert summary["total_input_tokens"] == 250
        assert summary["total_output_tokens"] == 500

    def test_budget_check(self):
        """Test budget checking."""
        tracker = CostTracker()

        # Should not raise for small amount
        tracker.track_request("openai", "gpt-4o-mini", 100, 100)
        tracker.check_budget(10.00)  # Should pass

        # Should raise for exceeding budget
        tracker.total_cost = 15.00
        with pytest.raises(Exception, match="budget.*exceeded"):
            tracker.check_budget(10.00)


class TestTokenCounting:
    """Test token counting."""

    def test_count_tokens(self):
        """Test token counting function."""
        text = "Hello, how are you?"
        count = count_tokens(text)

        assert count > 0
        assert count < len(text)  # Should be less than character count


class TestNewsAPI:
    """Test News API integration."""

    @patch('news_api.requests.get')
    def test_fetch_top_headlines(self, mock_get):
        """Test fetching headlines."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ok",
            "articles": [
                {
                    "title": "Test Article",
                    "description": "Test description",
                    "content": "Test content",
                    "url": "https://example.com",
                    "source": {"name": "Test Source"},
                    "publishedAt": "2026-01-19"
                }
            ]
        }
        mock_get.return_value = mock_response

        api = NewsAPI()
        articles = api.fetch_top_headlines(max_articles=1)

        assert len(articles) == 1
        assert articles[0]["title"] == "Test Article"
        assert articles[0]["source"] == "Test Source"


class TestLLMProviders:
    """Test LLM provider integration."""

    @patch('llm_providers.OpenAI')
    def test_ask_openai(self, mock_openai_class):
        """Test OpenAI integration."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        providers = LLMProviders()
        providers.openai_client = mock_client

        response = providers.ask_openai("Test prompt")

        assert response == "Test response"
        assert mock_client.chat.completions.create.called


class TestNewsSummarizer:
    """Test news summarizer."""

    def test_initialization(self):
        """Test summarizer initialization."""
        summarizer = NewsSummarizer()

        assert summarizer.news_api is not None
        assert summarizer.llm_providers is not None

    @patch.object(LLMProviders, 'ask_openai')
    @patch.object(LLMProviders, 'ask_anthropic')
    def test_summarize_article(self, mock_anthropic, mock_openai, tmp_path):
        """Test article summarization."""
        mock_openai.return_value = "Test summary"
        mock_anthropic.return_value = "Positive sentiment"

        summarizer = NewsSummarizer()
        summarizer.cache = ResponseCache(cache_dir=str(tmp_path))
        article = {
            "title": "Test Article",
            "description": "Test description",
            "content": "Test content",
            "url": "https://example.com",
            "source": "Test Source",
            "published_at": "2026-01-19"
        }

        result = summarizer.summarize_article(article)

        assert result["title"] == "Test Article"
        assert result["summary"] == "Test summary"
        assert result["sentiment"] == "Positive sentiment"
        assert mock_openai.called
        assert mock_anthropic.called


class TestCacheStats:
    """Test cache statistics tracking."""

    def test_initial_state(self):
        """Test initial stats are zero."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0

    def test_hit_recording(self):
        """Test hit recording updates stats."""
        stats = CacheStats()
        stats.hits = 3
        stats.misses = 1
        assert stats.hit_rate == 75.0

    def test_miss_recording(self):
        """Test miss-only scenario."""
        stats = CacheStats()
        stats.misses = 4
        assert stats.hit_rate == 0.0

    def test_hit_rate_math(self):
        """Test hit rate calculation."""
        stats = CacheStats()
        stats.hits = 1
        stats.misses = 1
        assert stats.hit_rate == 50.0


class TestResponseCache:
    """Test response cache."""

    def test_miss_returns_none(self, tmp_path):
        """Test cache miss returns None."""
        cache = ResponseCache(cache_dir=str(tmp_path))
        assert cache.get("https://example.com/article") is None
        assert cache.stats.misses == 1

    def test_set_and_get(self, tmp_path):
        """Test storing and retrieving a value."""
        cache = ResponseCache(cache_dir=str(tmp_path))
        data = {"summary": "Test summary", "sentiment": "positive"}
        cache.set("https://example.com/article", data)

        result = cache.get("https://example.com/article")
        assert result == data
        assert cache.stats.hits == 1

    def test_persistence_across_instances(self, tmp_path):
        """Test cache survives across instances."""
        cache1 = ResponseCache(cache_dir=str(tmp_path))
        cache1.set("https://example.com/article", {"summary": "cached"})

        cache2 = ResponseCache(cache_dir=str(tmp_path))
        result = cache2.get("https://example.com/article")
        assert result == {"summary": "cached"}

    def test_corrupt_file_recovery(self, tmp_path):
        """Test graceful handling of corrupt cache file."""
        cache_file = tmp_path / "response_cache.json"
        cache_file.write_text("not valid json{{{")

        cache = ResponseCache(cache_dir=str(tmp_path))
        assert cache.get("https://example.com") is None

    def test_clear(self, tmp_path):
        """Test clearing the cache."""
        cache = ResponseCache(cache_dir=str(tmp_path))
        cache.set("https://example.com/article", {"data": "value"})
        cache.clear()

        assert cache.get("https://example.com/article") is None

    def test_directory_creation(self, tmp_path):
        """Test cache creates directory if missing."""
        cache_dir = str(tmp_path / "nested" / "cache")
        cache = ResponseCache(cache_dir=cache_dir)
        cache.set("https://example.com", {"data": "value"})

        assert os.path.exists(cache_dir)


class TestNewsSummarizerCache:
    """Test cache integration in NewsSummarizer."""

    @patch.object(LLMProviders, 'ask_openai')
    @patch.object(LLMProviders, 'ask_anthropic')
    def test_summarize_article_uses_cache(self, mock_anthropic, mock_openai, tmp_path):
        """Test that second call for same URL skips LLM calls."""
        mock_openai.return_value = "Test summary"
        mock_anthropic.return_value = "Positive sentiment"

        summarizer = NewsSummarizer()
        summarizer.cache = ResponseCache(cache_dir=str(tmp_path))

        article = {
            "title": "Test Article",
            "description": "Test description",
            "content": "Test content",
            "url": "https://example.com/cached",
            "source": "Test Source",
            "published_at": "2026-01-19"
        }

        # First call - should use LLM
        result1 = summarizer.summarize_article(article)
        assert mock_openai.call_count == 1
        assert mock_anthropic.call_count == 1

        # Second call - should use cache
        result2 = summarizer.summarize_article(article)
        assert mock_openai.call_count == 1  # No additional calls
        assert mock_anthropic.call_count == 1  # No additional calls
        assert result2 == result1


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
