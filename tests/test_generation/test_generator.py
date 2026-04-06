"""Tests for LLM generators with mocked providers."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from src.config import AnthropicConfig, OllamaConfig
from src.extractors.base import MetadataArtifact
from src.generation.generator import (
    AnthropicProvider,
    Generator,
    LLMProvider,
    OllamaProvider,
)
from src.retrieval.semantic_search import SearchResult


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    """Create sample search results for testing."""
    artifact = MetadataArtifact(
        id="postgresql_public_table_sale_order_123",
        name="sale_order",
        type="table",
        source_type="database",
        language="postgresql",
        module="public",
        description="Stores sales orders",
        metadata={},
    )
    return [SearchResult(artifact=artifact, similarity=0.85, distance=0.15, rank=1)]


@pytest.fixture
def ollama_config() -> OllamaConfig:
    """Create Ollama configuration for testing."""
    return OllamaConfig(
        base_url="http://localhost:11434",
        model="qwen2.5-coder:7b-instruct",
        temperature=0.1,
        top_p=0.9,
        top_k=40,
        max_tokens=1024,
        timeout_seconds=60,
    )


@pytest.fixture
def anthropic_config() -> AnthropicConfig:
    """Create Anthropic configuration for testing."""
    return AnthropicConfig(
        model="claude-sonnet-4-5-20250929",
        temperature=0.1,
        max_tokens=2048,
        track_costs=True,
    )


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, response: str = "Mock response"):
        self.response = response
        self.call_count = 0
        self.last_prompt = None
        self.last_system_message = None

    def generate(
        self,
        prompt: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Mock generate method."""
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system_message = system_message
        return self.response

    def provider_name(self) -> str:
        """Get provider name."""
        return "mock:test"


class TestOllamaProvider:
    """Tests for Ollama provider."""

    @patch("requests.get")
    def test_ollama_provider_initialization(self, mock_get: Mock, ollama_config: OllamaConfig):
        """Test Ollama provider initialization."""
        # Mock successful connection check
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        provider = OllamaProvider(ollama_config)

        assert provider.config == ollama_config
        assert provider.base_url == ollama_config.base_url
        assert provider.model == ollama_config.model
        mock_get.assert_called_once()

    @patch("requests.get")
    @patch("requests.post")
    def test_ollama_provider_generate(
        self, mock_post: Mock, mock_get: Mock, ollama_config: OllamaConfig
    ):
        """Test Ollama provider text generation."""
        # Mock connection check
        mock_get_response = Mock()
        mock_get_response.raise_for_status = Mock()
        mock_get.return_value = mock_get_response

        # Mock generation response
        mock_post_response = Mock()
        mock_post_response.raise_for_status = Mock()
        mock_post_response.json.return_value = {
            "message": {"content": "Generated answer with citation [sale_order] (ID: test_id)."}
        }
        mock_post.return_value = mock_post_response

        provider = OllamaProvider(ollama_config)
        answer = provider.generate(
            prompt="What is the sale_order table?",
            system_message="You are a helpful assistant.",
            temperature=0.2,
            max_tokens=512,
        )

        assert "Generated answer" in answer
        assert "[sale_order]" in answer
        mock_post.assert_called_once()

        # Check request payload
        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]
        assert payload["model"] == ollama_config.model
        assert payload["options"]["temperature"] == 0.2
        assert payload["options"]["num_predict"] == 512
        assert len(payload["messages"]) == 2  # system + user

    @patch("requests.get")
    @patch("requests.post")
    def test_ollama_provider_timeout(
        self, mock_post: Mock, mock_get: Mock, ollama_config: OllamaConfig
    ):
        """Test Ollama provider timeout handling."""
        # Mock connection check
        mock_get_response = Mock()
        mock_get_response.raise_for_status = Mock()
        mock_get.return_value = mock_get_response

        # Mock timeout
        mock_post.side_effect = requests.exceptions.Timeout()

        provider = OllamaProvider(ollama_config)

        with pytest.raises(requests.exceptions.Timeout):
            provider.generate(prompt="Test")

    @patch("requests.get")
    @patch("requests.post")
    def test_ollama_provider_uses_defaults(
        self, mock_post: Mock, mock_get: Mock, ollama_config: OllamaConfig
    ):
        """Test that Ollama provider uses config defaults when params not specified."""
        # Mock connection check
        mock_get_response = Mock()
        mock_get_response.raise_for_status = Mock()
        mock_get.return_value = mock_get_response

        # Mock generation response
        mock_post_response = Mock()
        mock_post_response.raise_for_status = Mock()
        mock_post_response.json.return_value = {"message": {"content": "Answer"}}
        mock_post.return_value = mock_post_response

        provider = OllamaProvider(ollama_config)
        provider.generate(prompt="Test")  # No temperature/max_tokens specified

        # Check that config defaults were used
        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]
        assert payload["options"]["temperature"] == ollama_config.temperature
        assert payload["options"]["num_predict"] == ollama_config.max_tokens

    @patch("requests.get")
    def test_ollama_provider_name(self, mock_get: Mock, ollama_config: OllamaConfig):
        """Test Ollama provider name."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        provider = OllamaProvider(ollama_config)
        name = provider.provider_name()

        assert "ollama" in name
        assert ollama_config.model in name


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    @patch("anthropic.Anthropic")
    def test_anthropic_provider_initialization(
        self, mock_anthropic: Mock, anthropic_config: AnthropicConfig
    ):
        """Test Anthropic provider initialization."""
        provider = AnthropicProvider(anthropic_config, api_key="test_key")

        assert provider.config == anthropic_config
        assert provider.model == anthropic_config.model
        assert provider.total_input_tokens == 0
        assert provider.total_output_tokens == 0
        assert provider.total_cost_usd == 0.0
        mock_anthropic.assert_called_once_with(api_key="test_key")

    @patch("anthropic.Anthropic")
    def test_anthropic_provider_generate(
        self, mock_anthropic_class: Mock, anthropic_config: AnthropicConfig
    ):
        """Test Anthropic provider text generation."""
        # Mock Anthropic client
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Generated answer from Claude.")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(anthropic_config, api_key="test_key")
        answer = provider.generate(
            prompt="What is the sale_order table?",
            system_message="You are Claude.",
            temperature=0.3,
            max_tokens=1024,
        )

        assert "Generated answer from Claude" in answer
        mock_client.messages.create.assert_called_once()

        # Check usage tracking
        assert provider.total_input_tokens == 100
        assert provider.total_output_tokens == 50
        assert provider.total_cost_usd > 0

    @patch("anthropic.Anthropic")
    def test_anthropic_provider_cost_calculation(
        self, mock_anthropic_class: Mock, anthropic_config: AnthropicConfig
    ):
        """Test Anthropic cost calculation."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Answer")]
        mock_response.usage.input_tokens = 1_000_000  # 1M tokens
        mock_response.usage.output_tokens = 1_000_000  # 1M tokens
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(anthropic_config, api_key="test_key")
        provider.generate(prompt="Test")

        # For Sonnet: $3 input + $15 output = $18 per 1M tokens each
        expected_cost = 3.00 + 15.00
        assert provider.total_cost_usd == pytest.approx(expected_cost, rel=0.01)

    @patch("anthropic.Anthropic")
    def test_anthropic_provider_usage_stats(
        self, mock_anthropic_class: Mock, anthropic_config: AnthropicConfig
    ):
        """Test getting usage statistics."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Answer")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(anthropic_config, api_key="test_key")
        provider.generate(prompt="Test")

        stats = provider.get_usage_stats()

        assert stats["total_input_tokens"] == 100
        assert stats["total_output_tokens"] == 50
        assert "total_cost_usd" in stats

    @patch("anthropic.Anthropic")
    def test_anthropic_provider_name(
        self, mock_anthropic_class: Mock, anthropic_config: AnthropicConfig
    ):
        """Test Anthropic provider name."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        provider = AnthropicProvider(anthropic_config, api_key="test_key")
        name = provider.provider_name()

        assert "anthropic" in name
        assert anthropic_config.model in name


class TestGenerator:
    """Tests for the main Generator class."""

    def test_generator_initialization(self):
        """Test generator initialization with a provider."""
        mock_provider = MockLLMProvider()
        generator = Generator(mock_provider)

        assert generator.provider == mock_provider

    def test_generator_answer_generation(self, sample_search_results: list[SearchResult]):
        """Test generating an answer using the generator."""
        mock_provider = MockLLMProvider(
            response="The sale_order table [sale_order] (ID: postgresql_public_table_sale_order_123) stores orders."
        )
        generator = Generator(mock_provider)

        answer = generator.generate_answer(
            query="What is the sale_order table?",
            search_results=sample_search_results,
        )

        assert "sale_order" in answer
        assert mock_provider.call_count == 1
        assert mock_provider.last_prompt is not None
        assert "sale_order" in mock_provider.last_prompt

    def test_generator_uses_correct_system_message_for_ollama(
        self, sample_search_results: list[SearchResult]
    ):
        """Test that generator uses Ollama-specific system message."""
        mock_provider = MockLLMProvider()
        mock_provider.provider_name = lambda: "ollama:qwen"

        generator = Generator(mock_provider)
        generator.generate_answer("Test query", sample_search_results)

        assert mock_provider.last_system_message is not None
        assert "Qwen" in mock_provider.last_system_message

    def test_generator_uses_correct_system_message_for_anthropic(
        self, sample_search_results: list[SearchResult]
    ):
        """Test that generator uses Anthropic-specific system message."""
        mock_provider = MockLLMProvider()
        mock_provider.provider_name = lambda: "anthropic:claude"

        generator = Generator(mock_provider)
        generator.generate_answer("Test query", sample_search_results)

        assert mock_provider.last_system_message is not None
        assert "Claude" in mock_provider.last_system_message

    def test_generator_passes_temperature_and_max_tokens(
        self, sample_search_results: list[SearchResult]
    ):
        """Test that generator passes temperature and max_tokens to provider."""
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.generate.return_value = "Answer"
        mock_provider.provider_name.return_value = "mock:test"

        generator = Generator(mock_provider)
        generator.generate_answer(
            query="Test",
            search_results=sample_search_results,
            temperature=0.5,
            max_tokens=512,
        )

        mock_provider.generate.assert_called_once()
        call_kwargs = mock_provider.generate.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 512

    def test_generator_handles_provider_errors(self, sample_search_results: list[SearchResult]):
        """Test that generator propagates provider errors."""
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.generate.side_effect = RuntimeError("Provider error")
        mock_provider.provider_name.return_value = "mock:test"

        generator = Generator(mock_provider)

        with pytest.raises(RuntimeError, match="Provider error"):
            generator.generate_answer("Test", sample_search_results)

    def test_generator_get_provider_stats_anthropic(self, anthropic_config: AnthropicConfig):
        """Test getting provider stats for Anthropic provider."""
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_anthropic_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Answer")]
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50
            mock_client.messages.create.return_value = mock_response

            provider = AnthropicProvider(anthropic_config, api_key="test_key")
            generator = Generator(provider)

            # Generate to accumulate stats
            generator.generate_answer("Test", [])

            stats = generator.get_provider_stats()

            assert stats["total_input_tokens"] == 100
            assert stats["total_output_tokens"] == 50

    def test_generator_get_provider_stats_ollama(self, ollama_config: OllamaConfig):
        """Test that Ollama provider returns empty stats."""
        with patch("requests.get"), patch("requests.post"):
            provider = OllamaProvider(ollama_config)
            generator = Generator(provider)

            stats = generator.get_provider_stats()

            assert stats == {}  # Ollama doesn't track stats


class TestProviderProtocol:
    """Tests for LLMProvider protocol compliance."""

    def test_mock_provider_implements_protocol(self):
        """Test that mock provider implements LLMProvider protocol."""
        mock_provider = MockLLMProvider()

        # Should be recognized as LLMProvider
        assert isinstance(mock_provider, LLMProvider)

    def test_ollama_provider_implements_protocol(self, ollama_config: OllamaConfig):
        """Test that OllamaProvider implements LLMProvider protocol."""
        with patch("requests.get"):
            provider = OllamaProvider(ollama_config)
            assert isinstance(provider, LLMProvider)

    def test_anthropic_provider_implements_protocol(self, anthropic_config: AnthropicConfig):
        """Test that AnthropicProvider implements LLMProvider protocol."""
        with patch("anthropic.Anthropic"):
            provider = AnthropicProvider(anthropic_config, api_key="test_key")
            assert isinstance(provider, LLMProvider)
