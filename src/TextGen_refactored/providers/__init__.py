"""
Language model provider factory and imports.
"""
from typing import Dict, Optional, Any, Union
from .base import LanguageModelProvider, BaseLanguageModelProvider, GenerationResponse, FilterProvider, LanguageResponse, ProviderConfig
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .gemini_provider import GeminiProvider
from .groq_provider import GroqProvider


class ProviderFactory:
    """Factory for creating language model providers."""
    
    def __init__(self):
        """Initialize the provider factory."""
        self._providers = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "gemini": GeminiProvider,
            "groq": GroqProvider
        }

    def create_provider(
        self, 
        provider_type: str,
        model_name: str = None,
        api_key: str = None, 
        default_sample_length: int = 1024,
        config_sampling: dict = None,
        seed: int = None
    ) -> LanguageModelProvider:
        """Create a language model provider.
        
        Args:
            provider_type: Type of the provider (e.g., "openai", "anthropic", "gemini", "groq")
            model_name: Name of the model to use
            api_key: API key for the provider
            default_sample_length: Default length of text to sample
            config_sampling: Sampling parameters
            seed: Random seed for sampling
            
        Returns:
            An instance of a language model provider
            
        Raises:
            ValueError: If the provider type is not supported
        """
        if provider_type not in self._providers:
            raise ValueError(
                f"Unsupported provider type: {provider_type}. "
                f"Supported types are: {', '.join(self._providers.keys())}"
            )
        
        # Create provider config
        config = ProviderConfig(
            model=model_name,
            api_key=api_key,
            default_sample_length=default_sample_length,
            config_sampling=config_sampling or {"temp": 0.7, "prob": 0.9},
            seed=seed
        )
        
        # Create and return provider
        return self._providers[provider_type](config)


__all__ = [
    'LanguageModelProvider',
    'BaseLanguageModelProvider',
    'GenerationResponse',
    'FilterProvider',
    'LanguageResponse',
    'ProviderConfig',
    'OpenAIProvider',
    'AnthropicProvider',
    'GeminiProvider',
    'GroqProvider',
    'ProviderFactory'
] 