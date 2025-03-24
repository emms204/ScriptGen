"""
Groq API provider for TextGen.

This module implements the LanguageModelProvider interface for the Groq API,
which provides access to various open-source models like Llama, Mistral, etc.
"""
import os
import json
import time
from typing import Dict, List, Optional, Any, Union, Iterator

import requests
from requests.exceptions import RequestException

from .base import LanguageModelProvider, LanguageResponse, ProviderConfig


class GroqProvider(LanguageModelProvider):
    """Provider for Groq API."""

    # Model name mapping for commonly used models on Groq
    MODEL_MAPPING = {
        "llama3-8b": "llama3-8b-8192",
        "llama3-70b": "llama3-70b-8192",
        "llama3.1-8b": "llama3.1-8b-8192",
        "llama3.1-70b": "llama3.1-70b-8192",
        "mixtral-8x7b": "mixtral-8x7b-32768",
        "gemma-7b": "gemma-7b-it",
        "deepseek-llm": "deepseek-llm-67b-chat",
        "gemma2-9b": "gemma2-9b-it",
        "gemma2-27b": "gemma2-27b-it",
        "mistral-7b": "mistral-7b-instruct",
        "mixtral-8x22b": "mixtral-8x22b-32768",
        "qwen2-7b": "qwen2-7b-instruct"
    }

    def __init__(self, config: ProviderConfig):
        """Initialize Groq provider.
        
        Args:
            config: Configuration for the provider
        """
        super().__init__(config)
        
        # Set API key
        self.api_key = config.api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is required")
        
        # Set base URL for Groq API
        self.base_url = "https://api.groq.com/openai/v1"
        
        # Set model
        self.model = self._get_model_name(config.model)

    def _get_model_name(self, model_name: str) -> str:
        """Get the full model name for Groq API.
        
        Args:
            model_name: The model name as used in TextGen
            
        Returns:
            The full model name as used in Groq API
        """
        return self.MODEL_MAPPING.get(model_name, model_name)

    def sample(
        self,
        prompt: str,
        sample_length: Optional[int] = None,
        seed: Optional[int] = None,
        num_samples: int = 1,
        stream: bool = False
    ) -> Union[List[LanguageResponse], Iterator[str]]:
        """Generate samples from the model.
        
        Args:
            prompt: The prompt to use
            sample_length: The maximum number of tokens to generate
            seed: Random seed for generation
            num_samples: Number of samples to generate
            stream: Whether to stream the response
            
        Returns:
            A list of LanguageResponse objects or an iterator of response chunks
        """
        # Override None with default sample length
        if sample_length is None:
            sample_length = self.default_sample_length
        
        # Set up headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Set up request data
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": sample_length,
            "temperature": self.config_sampling.get("temp", 0.7),
            "top_p": self.config_sampling.get("prob", 0.9),
            "stream": stream
        }
        
        # Add seed if provided
        if seed is not None:
            data["seed"] = seed
        
        # Set up endpoint URL
        endpoint = f"{self.base_url}/chat/completions"
        
        try:
            if stream:
                return self._stream_response(endpoint, headers, data)
            else:
                return self._batch_response(endpoint, headers, data, num_samples, prompt)
        except RequestException as e:
            raise RuntimeError(f"Error calling Groq API: {str(e)}")

    def _batch_response(
        self, 
        endpoint: str, 
        headers: Dict[str, str], 
        data: Dict[str, Any], 
        num_samples: int, 
        prompt: str
    ) -> List[LanguageResponse]:
        """Get batch responses from the API.
        
        Args:
            endpoint: API endpoint
            headers: Request headers
            data: Request data
            num_samples: Number of samples to generate
            prompt: Original prompt
            
        Returns:
            List of LanguageResponse objects
        """
        responses = []
        
        # Make multiple requests if num_samples > 1
        for _ in range(num_samples):
            response = requests.post(endpoint, headers=headers, json=data)
            response.raise_for_status()
            response_json = response.json()
            
            # Extract the generated text
            text = response_json["choices"][0]["message"]["content"]
            
            responses.append(
                LanguageResponse(
                    prompt=prompt,
                    prompt_length=len(prompt),
                    text=text,
                    text_length=len(text)
                )
            )
        
        return responses

    def _stream_response(self, endpoint: str, headers: Dict[str, str], data: Dict[str, Any]) -> Iterator[str]:
        """Stream responses from the API.
        
        Args:
            endpoint: API endpoint
            headers: Request headers
            data: Request data
            
        Returns:
            Iterator of response chunks
        """
        response = requests.post(endpoint, headers=headers, json=data, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                
                # Skip empty lines and "[DONE]"
                if line == "[DONE]" or not line.strip():
                    continue
                
                # Lines in SSE always start with "data: "
                if line.startswith("data: "):
                    line = line[6:]  # Remove "data: " prefix
                    
                    try:
                        chunk_data = json.loads(line)
                        delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue 