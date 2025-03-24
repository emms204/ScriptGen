import os
import json
import time
import warnings
from typing import Dict, List, Any, Optional, Union
import logging

# Try to import requests, but provide a fallback if not available
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    warnings.warn(
        "Requests package not found. HTTP functionality will be unavailable. "
        "Install requests with 'pip install requests' for full functionality."
    )
    REQUESTS_AVAILABLE = False
    
    # Create a simple mock for requests
    class MockResponse:
        def __init__(self, status_code=200, data=None):
            self.status_code = status_code
            self._data = data or {}
            
        def json(self):
            return self._data
            
        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception(f"HTTP Error: {self.status_code}")
    
    class requests:
        @staticmethod
        def post(*args, **kwargs):
            warnings.warn("Using mock requests.post - no actual HTTP request will be made")
            return MockResponse(data={"text": "Mock LLM response - Requests package not available"})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('llm_wrapper')

class LLMWrapper:
    """
    A standardized interface to interact with various LLMs
    (OpenAI, Grok, llama, mistral, deepseek, qwen).
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the LLM wrapper with configuration.
        
        Args:
            config_path: Path to the config file (JSON)
        """
        self.config = self._load_config(config_path)
        self.default_model = self.config.get('default_model', 'gpt-3.5-turbo')
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from a JSON file or environment variables.
        
        Args:
            config_path: Path to the config file
            
        Returns:
            Dictionary containing configuration
        """
        config = {
            'models': {},
            'default_model': 'gpt-3.5-turbo'
        }
        
        # If no config path is provided, use the default
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'config',
                'models.json'
            )
        
        # Try to load from file if provided
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config)
                logger.info(f"Loaded config from {config_path}")
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {str(e)}")
        
        # Check for environment variables as fallback
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if openai_api_key:
            if 'models' not in config:
                config['models'] = {}
                
            if 'openai' not in config['models']:
                config['models']['openai'] = {}
                
            config['models']['openai']['api_key'] = openai_api_key
            
            # Add default models if they don't exist
            if 'available_models' not in config['models']['openai']:
                config['models']['openai']['available_models'] = [
                    'gpt-3.5-turbo',
                    'gpt-4',
                    'gpt-4-turbo'
                ]
        
        return config
    
    def call_llm(
        self, 
        prompt: str, 
        model: Optional[str] = None, 
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        retry_delay: int = 2
    ) -> Dict[str, Any]:
        """
        Send a prompt to the chosen LLM and return the response.
        
        Args:
            prompt: The prompt to send to the LLM
            model: The model to use (default is from config)
            params: Additional parameters for the API call
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
            
        Returns:
            Dictionary containing the LLM response
        """
        model = model or self.default_model
        params = params or {}
        
        # Determine provider from model name
        provider = self._get_provider_for_model(model)
        
        if not provider:
            raise ValueError(f"Unknown model: {model}. No provider found.")
        
        # Set default parameters
        default_params = {
            'max_tokens': 500,
            'temperature': 0.7
        }
        
        # Merge with user-provided params
        call_params = {**default_params, **params}
        
        # Execute with retry logic
        for attempt in range(max_retries):
            try:
                if provider == 'openai':
                    return self._call_openai(prompt, model, call_params)
                elif provider == 'mistral':
                    return self._call_mistral(prompt, model, call_params)
                elif provider == 'local':
                    return self._call_local(prompt, model, call_params)
                else:
                    raise ValueError(f"Provider {provider} not implemented")
            except Exception as e:
                if 'rate limit' in str(e).lower() and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limit hit. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                elif attempt < max_retries - 1:
                    logger.warning(f"Error calling LLM: {str(e)}. Retrying...")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Failed to call LLM after {max_retries} attempts: {str(e)}")
                    raise
    
    def _get_provider_for_model(self, model: str) -> Optional[str]:
        """
        Determine which provider to use based on the model name.
        
        Args:
            model: The model name
            
        Returns:
            Provider name or None if not found
        """
        # Check each provider's available models
        for provider, config in self.config.get('models', {}).items():
            if model in config.get('available_models', []):
                return provider
                
        # Try to infer from model name as fallback
        if model.startswith('gpt-'):
            return 'openai'
        elif model.startswith('mistral-'):
            return 'mistral'
        elif any(name in model.lower() for name in ['llama', 'deepseek', 'qwen']):
            return 'local'
            
        return None
    
    def _call_openai(self, prompt: str, model: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the OpenAI API.
        
        Args:
            prompt: The prompt to send
            model: The model to use
            params: Additional parameters
            
        Returns:
            The API response
        """
        try:
            import openai
            OPENAI_AVAILABLE = True
        except ImportError:
            OPENAI_AVAILABLE = False
            warnings.warn("OpenAI package not installed. Using mock response. Please install with 'pip install openai'")
            return {
                'text': f"Mock response for '{prompt[:30]}...' - OpenAI package not installed",
                'model': model,
                'provider': 'openai',
                'raw_response': {}
            }
        
        # Get API key from config
        api_key = self.config.get('models', {}).get('openai', {}).get('api_key')
        if not api_key:
            warnings.warn("OpenAI API key not found in config. Using mock response.")
            return {
                'text': f"Mock response for '{prompt[:30]}...' - API key not configured",
                'model': model,
                'provider': 'openai',
                'raw_response': {}
            }
        
        openai.api_key = api_key
        
        messages = [{"role": "user", "content": prompt}]
        
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=params.get('max_tokens', 500),
            temperature=params.get('temperature', 0.7),
            top_p=params.get('top_p', 1.0),
            frequency_penalty=params.get('frequency_penalty', 0.0),
            presence_penalty=params.get('presence_penalty', 0.0)
        )
        
        return {
            'text': response['choices'][0]['message']['content'],
            'model': model,
            'provider': 'openai',
            'raw_response': response
        }
    
    def _call_mistral(self, prompt: str, model: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the Mistral AI API.
        
        Args:
            prompt: The prompt to send
            model: The model to use
            params: Additional parameters
            
        Returns:
            The API response
        """
        try:
            from mistralai.client import MistralClient
            from mistralai.models.chat_completion import ChatMessage
            MISTRAL_AVAILABLE = True
        except ImportError:
            MISTRAL_AVAILABLE = False
            warnings.warn("Mistral AI package not installed. Using mock response. Please install with 'pip install mistralai'")
            return {
                'text': f"Mock response for '{prompt[:30]}...' - Mistral AI package not installed",
                'model': model,
                'provider': 'mistral',
                'raw_response': {}
            }
        
        # Get API key from config
        api_key = self.config.get('models', {}).get('mistral', {}).get('api_key')
        if not api_key:
            warnings.warn("Mistral API key not found in config. Using mock response.")
            return {
                'text': f"Mock response for '{prompt[:30]}...' - API key not configured",
                'model': model,
                'provider': 'mistral',
                'raw_response': {}
            }
        
        client = MistralClient(api_key=api_key)
        
        messages = [ChatMessage(role="user", content=prompt)]
        
        response = client.chat(
            model=model,
            messages=messages,
            max_tokens=params.get('max_tokens', 500),
            temperature=params.get('temperature', 0.7),
            top_p=params.get('top_p', 1.0)
        )
        
        return {
            'text': response.choices[0].message.content,
            'model': model,
            'provider': 'mistral',
            'raw_response': response
        }
    
    def _call_local(self, prompt: str, model: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a local model (llama, deepseek, qwen) via API.
        
        Args:
            prompt: The prompt to send
            model: The model to use
            params: Additional parameters
            
        Returns:
            The API response
        """
        if not REQUESTS_AVAILABLE:
            warnings.warn("Requests package not installed. Using mock response.")
            return {
                'text': f"Mock response for '{prompt[:30]}...' - Requests package not installed",
                'model': model,
                'provider': 'local',
                'raw_response': {}
            }
            
        # Get endpoint from config
        endpoint = self.config.get('models', {}).get('local', {}).get('endpoint')
        if not endpoint:
            warnings.warn("Local model endpoint not found in config. Using mock response.")
            return {
                'text': f"Mock response for '{prompt[:30]}...' - Endpoint not configured",
                'model': model,
                'provider': 'local',
                'raw_response': {}
            }
        
        payload = {
            'model': model,
            'prompt': prompt,
            'max_tokens': params.get('max_tokens', 500),
            'temperature': params.get('temperature', 0.7),
            'top_p': params.get('top_p', 1.0)
        }
        
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        return {
            'text': result.get('text', ''),
            'model': model,
            'provider': 'local',
            'raw_response': result
        }
    
    def parse_response(self, response: Dict[str, Any], format_type: str = 'text') -> Union[str, Dict, List]:
        """
        Extract structured output from the LLM response.
        
        Args:
            response: The LLM response to parse
            format_type: Type of format to extract ('text', 'json', 'lines', 'dialogue')
            
        Returns:
            Parsed content in the requested format
        """
        if not response or 'text' not in response:
            raise ValueError("Invalid response format")
            
        text = response['text'].strip()
        
        if format_type == 'text':
            return text
            
        elif format_type == 'json':
            # Try to extract JSON from the response
            try:
                # Check if the entire response is valid JSON
                return json.loads(text)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                import re
                json_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
                
                if json_blocks:
                    for block in json_blocks:
                        try:
                            return json.loads(block.strip())
                        except json.JSONDecodeError:
                            continue
                
                # If no valid JSON found, raise error
                raise ValueError("Could not parse JSON from response")
                
        elif format_type == 'lines':
            # Split text into non-empty lines
            return [line.strip() for line in text.split('\n') if line.strip()]
            
        elif format_type == 'dialogue':
            # Extract dialogue lines in format "Character: Line"
            import re
            dialogue_lines = []
            
            for line in text.split('\n'):
                match = re.match(r'^([\w\s]+?):\s*(.+)$', line.strip())
                if match:
                    character, text = match.groups()
                    dialogue_lines.append({
                        'character': character.strip(),
                        'text': text.strip()
                    })
            
            return dialogue_lines
            
        else:
            raise ValueError(f"Unknown format type: {format_type}")
            
    def get_available_models(self) -> List[str]:
        """
        Get a list of all available models across providers.
        
        Returns:
            List of available model names
        """
        models = []
        
        for provider, config in self.config.get('models', {}).items():
            models.extend(config.get('available_models', []))
            
        return models 