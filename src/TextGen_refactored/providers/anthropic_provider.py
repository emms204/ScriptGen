"""
Anthropic Claude language model provider implementation.
"""
import os
from typing import Dict, List, Optional, Any
import anthropic
from .base import BaseLanguageModelProvider, GenerationResponse


class AnthropicProvider(BaseLanguageModelProvider):
    """Provider for Anthropic Claude language models."""
    
    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20240620",
        api_key: Optional[str] = None,
        system_prompt: str = "You are a helpful playwright assistant.",
        default_sample_length: int = 2048,
        config_sampling: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        max_retries: int = 3,
        timeout: float = 120.0
    ):
        """Initialize the Anthropic Claude provider.
        
        Args:
            model_name: Name of the Anthropic model to use
            api_key: Anthropic API key (if None, uses environment variable)
            system_prompt: System prompt/instruction
            default_sample_length: Default length for generation
            config_sampling: Configuration for sampling
            seed: Random seed for reproducibility
            max_retries: Maximum number of retries for failed requests
            timeout: Timeout for requests in seconds
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
        
        # Initialize the Anthropic client
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("SONNET_API_KEY")
        )
    
    def sample(
        self,
        prompt: str,
        sample_length: Optional[int] = None,
        temperature: float = 0.5,
        seed: Optional[int] = None,
        num_samples: int = 1
    ) -> List[GenerationResponse]:
        """Generate text samples from the Anthropic Claude model.
        
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
        
        results = []
        for _ in range(num_samples):
            # Create the API request
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=sample_length,
                temperature=temperature,
                top_p=self.config_sampling["prob"],
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract the text from the response
            response_text = response.content[0].text if response.content else ""
            
            results.append(
                GenerationResponse(
                    prompt=prompt,
                    prompt_length=len(prompt),
                    text=response_text,
                    text_length=len(response_text)
                )
            )
        
        return results 