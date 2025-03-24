"""
OpenAI language model provider implementation.
"""
import os
from typing import Dict, List, Optional, Any
from openai import OpenAI
from .base import BaseLanguageModelProvider, GenerationResponse


class OpenAIProvider(BaseLanguageModelProvider):
    """Provider for OpenAI language models."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        system_prompt: str = "You are a helpful playwright assistant.",
        default_sample_length: int = 1024,
        config_sampling: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        max_retries: int = 3,
        timeout: float = 120.0,
        frequency_penalty: float = 0.2,
        presence_penalty: float = 0.2
    ):
        """Initialize the OpenAI provider.
        
        Args:
            model_name: Name of the OpenAI model to use
            api_key: OpenAI API key (if None, uses environment variable)
            system_prompt: System prompt/instruction
            default_sample_length: Default length for generation
            config_sampling: Configuration for sampling
            seed: Random seed for reproducibility
            max_retries: Maximum number of retries for failed requests
            timeout: Timeout for requests in seconds
            frequency_penalty: Frequency penalty for generation
            presence_penalty: Presence penalty for generation
        """
        if config_sampling is None:
            config_sampling = {"prob": 0.9, "temp": 0.7}
        
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            default_sample_length=default_sample_length,
            config_sampling=config_sampling,
            seed=seed,
            max_retries=max_retries,
            timeout=timeout
        )
        
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        
        # Initialize the OpenAI client
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    
    def sample(
        self,
        prompt: str,
        sample_length: Optional[int] = None,
        temperature: float = 0.5,
        seed: Optional[int] = None,
        num_samples: int = 1
    ) -> List[GenerationResponse]:
        """Generate text samples from the OpenAI model.
        
        Args:
            prompt: The prompt text to generate from
            sample_length: Maximum length of the generated sample
            temperature: Sampling temperature (higher = more random)
            seed: Random seed for reproducibility
            num_samples: Number of samples to generate
            
        Returns:
            List of generation responses
        """
        if sample_length is None:
            sample_length = self.default_sample_length
        
        if seed is None:
            seed = self.seed
        
        # Create the API request
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=sample_length,
            temperature=temperature,
            top_p=self.config_sampling["prob"],
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            seed=seed,
            n=num_samples,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the text from the response
        results = []
        for choice in response.choices:
            results.append(
                GenerationResponse(
                    prompt=prompt,
                    prompt_length=len(prompt),
                    text=choice.message.content or "",
                    text_length=len(choice.message.content or "")
                )
            )
        
        return results 