"""
Base classes for language model providers.

This module provides the foundational classes for interacting with various language models.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union

@dataclass
class GenerationResponse:
    """Response from a language model generation request."""
    prompt: str
    prompt_length: int
    text: str
    text_length: int


class LanguageModelProvider(ABC):
    """Abstract base class for language model providers."""
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the model."""
        pass
    
    @property
    @abstractmethod
    def default_sample_length(self) -> int:
        """Get the default sample length for the model."""
        pass
    
    @property
    @abstractmethod
    def config_sampling(self) -> Dict[str, Any]:
        """Get the sampling configuration for the model."""
        pass
    
    @abstractmethod
    def sample(
        self,
        prompt: str,
        sample_length: Optional[int] = None,
        temperature: float = 0.5,
        seed: Optional[int] = None,
        num_samples: int = 1
    ) -> List[GenerationResponse]:
        """Generate text samples from the model.
        
        Args:
            prompt: The prompt text to generate from
            sample_length: Maximum length of the generated sample
            temperature: Sampling temperature (higher = more random)
            seed: Random seed for reproducibility
            num_samples: Number of samples to generate
            
        Returns:
            List of generation responses
        """
        pass


class FilterProvider(ABC):
    """Abstract base class for content filtering."""
    
    @abstractmethod
    def validate(self, text: str) -> bool:
        """Validate whether the provided text is acceptable.
        
        Args:
            text: The text to validate
            
        Returns:
            True if the text is acceptable, False otherwise
        """
        pass


class BaseLanguageModelProvider(LanguageModelProvider):
    """Base implementation for language model providers with common functionality."""
    
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        default_sample_length: int,
        config_sampling: Dict[str, Any],
        seed: Optional[int] = None,
        max_retries: int = 3,
        timeout: float = 120.0
    ):
        """Initialize the language model provider.
        
        Args:
            model_name: Name of the model
            system_prompt: System prompt/instruction
            default_sample_length: Default length for generation
            config_sampling: Configuration for sampling
            seed: Random seed for reproducibility
            max_retries: Maximum number of retries for failed requests
            timeout: Timeout for requests in seconds
        """
        self._model_name = model_name
        self._system_prompt = system_prompt
        self._default_sample_length = default_sample_length
        self._config_sampling = config_sampling
        self._seed = seed
        self._max_retries = max_retries
        self._timeout = timeout
    
    @property
    def model_name(self) -> str:
        """Get the name of the model."""
        return self._model_name
    
    @property
    def system_prompt(self) -> str:
        """Get the system prompt for the model."""
        return self._system_prompt
    
    @property
    def default_sample_length(self) -> int:
        """Get the default sample length for the model."""
        return self._default_sample_length
    
    @property
    def config_sampling(self) -> Dict[str, Any]:
        """Get the sampling configuration for the model."""
        return self._config_sampling
    
    @property
    def seed(self) -> Optional[int]:
        """Get the random seed for reproducibility."""
        return self._seed
    
    @property
    def max_retries(self) -> int:
        """Get the maximum number of retries for failed requests."""
        return self._max_retries
    
    @property
    def timeout(self) -> float:
        """Get the timeout for requests in seconds."""
        return self._timeout 