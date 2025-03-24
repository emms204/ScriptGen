"""
Google Gemini language model provider implementation.
"""
import os
from typing import Dict, List, Optional, Any
import google.generativeai as genai
from .base import BaseLanguageModelProvider, GenerationResponse


class GeminiProvider(BaseLanguageModelProvider):
    """Provider for Google Gemini language models."""
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        system_prompt: str = "You are a creative writing assistant for a team of writers. Your goal is to expand on the input text prompt and to generate the continuation of that text without any comments. Be as creative as possible, write rich detailed descriptions and use precise language. Add new original ideas.",
        default_sample_length: int = 1024,
        config_sampling: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        max_retries: int = 3,
        timeout: float = 120.0
    ):
        """Initialize the Google Gemini provider.
        
        Args:
            model_name: Name of the Gemini model to use
            api_key: Google API key (if None, uses environment variable)
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
        
        # Configure the Gemini API
        genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        
        # Initialize the Gemini client
        self.client = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt
        )
    
    def sample(
        self,
        prompt: str,
        sample_length: Optional[int] = None,
        temperature: float = 0.5,
        seed: Optional[int] = None,
        num_samples: int = 1
    ) -> List[GenerationResponse]:
        """Generate text samples from the Gemini model.
        
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
        
        messages = [
            {"role": "user", "parts": prompt}
        ]
        
        # Generate the content
        response = self.client.generate_content(
            messages,
            generation_config=genai.types.GenerationConfig(
                candidate_count=num_samples,
                max_output_tokens=sample_length,
                temperature=temperature,
                top_p=self.config_sampling["prob"]
            )
        )
        
        # Extract the text from the response
        results = []
        
        if hasattr(response, 'candidates'):
            for candidate in response.candidates:
                response_text = candidate.content.parts[0].text if candidate.content.parts else ""
                results.append(
                    GenerationResponse(
                        prompt=prompt,
                        prompt_length=len(prompt),
                        text=response_text,
                        text_length=len(response_text)
                    )
                )
        else:
            # Handle case where response might have a different structure
            response_text = response.text if hasattr(response, 'text') and response.text else ""
            results.append(
                GenerationResponse(
                    prompt=prompt,
                    prompt_length=len(prompt),
                    text=response_text,
                    text_length=len(response_text)
                )
            )
        
        return results 