"""LLM-powered answer generation with multiple provider support."""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

import anthropic
import requests

from src.config import AnthropicConfig, OllamaConfig
from src.generation.prompts import get_system_message, build_qa_prompt
from src.observability import (
    StructuredLogger,
    get_correlation_id,
    record_error,
    record_operation,
    trace_function,
)
from src.retrieval.semantic_search import SearchResult

logger = StructuredLogger("cheap_rag.generation")


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers (structural typing)."""

    def generate(
        self,
        prompt: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: User prompt.
            system_message: System message/instructions.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text.
        """
        ...

    def provider_name(self) -> str:
        """Get provider name for logging."""
        ...


class OllamaProvider:
    """Ollama LLM provider for local inference."""

    def __init__(self, config: OllamaConfig):
        """Initialize Ollama provider.

        Args:
            config: Ollama configuration.
        """
        self.config = config
        self.base_url = config.base_url
        self.model = config.model

        logger.info("Initialized Ollama provider", model=self.model, base_url=self.base_url)

        # Verify Ollama is running
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info("Ollama server is reachable")
        except Exception as e:
            logger.warning("Ollama server may not be running", error=str(e))

    def generate(
        self,
        prompt: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text using Ollama API.

        Args:
            prompt: User prompt.
            system_message: System message.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.

        Returns:
            Generated text.
        """
        # Use config defaults if not specified
        if temperature is None:
            temperature = self.config.temperature
        if max_tokens is None:
            max_tokens = self.config.max_tokens

        # Build messages
        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        # Build request
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "num_predict": max_tokens,
            },
        }

        logger.debug("Ollama request", model=self.model, temperature=temperature)

        start_time = time.perf_counter()

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()

            duration_ms = (time.perf_counter() - start_time) * 1000
            result = response.json()

            answer = result.get("message", {}).get("content", "")
            record_operation(
                "llm_generate",
                duration_ms,
                {"provider": "ollama", "model": self.model},
            )
            logger.info(
                "Ollama generation completed",
                duration_ms=round(duration_ms, 1),
                model=self.model,
            )

            return answer.strip()

        except requests.exceptions.Timeout:
            record_error(
                "generation",
                requests.exceptions.Timeout(
                    f"Ollama request timed out after {self.config.timeout_seconds}s"
                ),
                get_correlation_id(),
            )
            raise
        except requests.exceptions.RequestException as e:
            record_error("generation", e, get_correlation_id())
            raise
        except Exception as e:
            record_error("generation", e, get_correlation_id())
            raise

    def provider_name(self) -> str:
        """Get provider name."""
        return f"ollama:{self.model}"


class AnthropicProvider:
    """Anthropic Claude provider for API-based inference."""

    def __init__(self, config: AnthropicConfig, api_key: str):
        """Initialize Anthropic provider.

        Args:
            config: Anthropic configuration.
            api_key: Anthropic API key.
        """
        self.config = config
        self.model = config.model

        logger.info("Initialized Anthropic provider", model=self.model)

        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=api_key)

        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

    def generate(
        self,
        prompt: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text using Anthropic API.

        Args:
            prompt: User prompt.
            system_message: System message.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.

        Returns:
            Generated text.
        """
        # Use config defaults if not specified
        if temperature is None:
            temperature = self.config.temperature
        if max_tokens is None:
            max_tokens = self.config.max_tokens

        logger.debug("Anthropic request", model=self.model, temperature=temperature)

        start_time = time.perf_counter()

        try:
            # Build message request
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }

            if system_message:
                kwargs["system"] = system_message

            response = self.client.messages.create(**kwargs)  # type: ignore[reportUnknownVariableType]  # anthropic

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Extract answer
            answer = response.content[0].text  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]  # anthropic

            # Track usage
            input_tokens = response.usage.input_tokens  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]  # anthropic
            output_tokens = response.usage.output_tokens  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]  # anthropic

            self.total_input_tokens += input_tokens  # type: ignore[reportUnknownMemberType]  # anthropic
            self.total_output_tokens += output_tokens  # type: ignore[reportUnknownMemberType]  # anthropic

            # Calculate cost (approximate pricing)
            cost = self._calculate_cost(input_tokens, output_tokens)  # type: ignore[reportUnknownArgumentType]  # anthropic
            self.total_cost_usd += cost

            record_operation(
                "llm_generate",
                duration_ms,
                {
                    "provider": "anthropic",
                    "model": self.model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": cost,
                },
            )
            logger.info(
                "Anthropic generation completed",
                duration_ms=round(duration_ms, 1),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            if self.config.track_costs:
                logger.info(
                    "Cost tracking", cost_usd=f"{cost:.4f}", total_usd=f"{self.total_cost_usd:.4f}"
                )

            return answer.strip()  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]  # anthropic

        except anthropic.APIError as e:
            record_error("generation", e, get_correlation_id())
            raise
        except Exception as e:
            record_error("generation", e, get_correlation_id())
            raise

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate approximate cost in USD.

        Pricing as of 2025:
        - Sonnet 4.5: $3.00 / 1M input, $15.00 / 1M output
        - Haiku 4.5: $0.80 / 1M input, $4.00 / 1M output

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Cost in USD.
        """
        # Pricing per 1M tokens
        if "sonnet" in self.model.lower():
            input_cost_per_1m = 3.00
            output_cost_per_1m = 15.00
        elif "haiku" in self.model.lower():
            input_cost_per_1m = 0.80
            output_cost_per_1m = 4.00
        else:
            # Default to Sonnet pricing
            input_cost_per_1m = 3.00
            output_cost_per_1m = 15.00

        cost = (input_tokens / 1_000_000) * input_cost_per_1m + (
            output_tokens / 1_000_000
        ) * output_cost_per_1m

        return cost

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics.

        Returns:
            Dictionary with token counts and costs.
        """
        return {
            "total_input_tokens": self.total_input_tokens,  # type: ignore[reportUnknownMemberType]  # anthropic
            "total_output_tokens": self.total_output_tokens,  # type: ignore[reportUnknownMemberType]  # anthropic
            "total_cost_usd": self.total_cost_usd,
        }

    def provider_name(self) -> str:
        """Get provider name."""
        return f"anthropic:{self.model}"


class Generator:
    """Main generator class with provider abstraction."""

    def __init__(self, provider: LLMProvider):
        """Initialize generator.

        Args:
            provider: LLM provider instance.
        """
        self.provider = provider
        logger.info("Generator initialized", provider=provider.provider_name())

    @trace_function("generate_answer")
    def generate_answer(
        self,
        query: str,
        search_results: list[SearchResult],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate answer to query using retrieved context.

        Args:
            query: User's question.
            search_results: Retrieved metadata artifacts.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.

        Returns:
            Generated answer with citations.
        """
        logger.info(
            "Generating answer",
            query=query,
            num_artifacts=len(search_results),
        )

        # Build prompt from query and search results
        prompt = build_qa_prompt(query, search_results)

        # Get system message for provider
        provider_name = self.provider.provider_name()
        if "anthropic" in provider_name:
            system_message = get_system_message("anthropic")
        elif "ollama" in provider_name:
            system_message = get_system_message("ollama")
        else:
            system_message = get_system_message("base")

        # Generate answer
        start = time.perf_counter()
        try:
            answer = self.provider.generate(
                prompt=prompt,
                system_message=system_message,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            duration_ms = (time.perf_counter() - start) * 1000
            record_operation(
                "generate_answer",
                duration_ms,
                {"query_length": len(query), "num_artifacts": len(search_results)},
            )
            logger.info("Answer generated successfully", duration_ms=round(duration_ms, 1))
            return answer

        except Exception as e:
            record_error("generation", e, get_correlation_id())
            raise

    def get_provider_stats(self) -> dict[str, Any]:
        """Get provider statistics (if available).

        Returns:
            Provider stats dictionary.
        """
        if isinstance(self.provider, AnthropicProvider):
            return self.provider.get_usage_stats()
        return {}
