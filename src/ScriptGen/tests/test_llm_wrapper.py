import unittest
import json
import os
from unittest.mock import patch, MagicMock
from ..llm.llm_wrapper import LLMWrapper

class TestLLMWrapper(unittest.TestCase):
    """Test cases for the LLMWrapper module"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary config file
        self.config_path = 'test_models.json'
        config = {
            "default_model": "gpt-test",
            "models": {
                "openai": {
                    "api_key": "test-key",
                    "available_models": ["gpt-test"]
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f)
            
        self.llm_wrapper = LLMWrapper(config_path=self.config_path)
    
    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
    
    def test_load_config(self):
        """Test loading configuration"""
        # Verify config was loaded correctly
        self.assertEqual(self.llm_wrapper.default_model, "gpt-test")
        self.assertEqual(
            self.llm_wrapper.config['models']['openai']['api_key'],
            "test-key"
        )
    
    @patch('openai.ChatCompletion.create')
    def test_call_openai(self, mock_create):
        """Test calling OpenAI API"""
        # Mock response
        mock_response = {
            'choices': [
                {
                    'message': {
                        'content': 'This is a test response.'
                    }
                }
            ]
        }
        mock_create.return_value = mock_response
        
        # Call LLM
        response = self.llm_wrapper.call_llm(
            prompt="Test prompt",
            model="gpt-test"
        )
        
        # Verify response
        self.assertEqual(response['text'], 'This is a test response.')
        self.assertEqual(response['provider'], 'openai')
        self.assertEqual(response['model'], 'gpt-test')
    
    def test_parse_response_text(self):
        """Test parsing text response"""
        response = {
            'text': 'This is a test response.'
        }
        
        result = self.llm_wrapper.parse_response(response, format_type='text')
        self.assertEqual(result, 'This is a test response.')
    
    def test_parse_response_json(self):
        """Test parsing JSON response"""
        json_data = {'name': 'Theo', 'occupation': 'Singer'}
        response = {
            'text': json.dumps(json_data)
        }
        
        result = self.llm_wrapper.parse_response(response, format_type='json')
        self.assertEqual(result, json_data)
    
    def test_parse_response_json_from_codeblock(self):
        """Test parsing JSON from a code block"""
        json_data = {'name': 'Theo', 'occupation': 'Singer'}
        response = {
            'text': f"Here's the data:\n\n```json\n{json.dumps(json_data)}\n```"
        }
        
        result = self.llm_wrapper.parse_response(response, format_type='json')
        self.assertEqual(result, json_data)
    
    def test_parse_response_lines(self):
        """Test parsing lines response"""
        response = {
            'text': "Line 1\nLine 2\n\nLine 3"
        }
        
        result = self.llm_wrapper.parse_response(response, format_type='lines')
        self.assertEqual(result, ['Line 1', 'Line 2', 'Line 3'])
    
    def test_parse_response_dialogue(self):
        """Test parsing dialogue response"""
        response = {
            'text': "Theo: I can't believe what happened.\nMaria: I told you to be careful."
        }
        
        result = self.llm_wrapper.parse_response(response, format_type='dialogue')
        expected = [
            {'character': 'Theo', 'text': "I can't believe what happened."},
            {'character': 'Maria', 'text': 'I told you to be careful.'}
        ]
        self.assertEqual(result, expected)
    
    def test_get_provider_for_model(self):
        """Test provider detection from model name"""
        # Test from available models
        provider = self.llm_wrapper._get_provider_for_model('gpt-test')
        self.assertEqual(provider, 'openai')
        
        # Test inference from name
        provider = self.llm_wrapper._get_provider_for_model('gpt-4')
        self.assertEqual(provider, 'openai')
        
        provider = self.llm_wrapper._get_provider_for_model('mistral-medium')
        self.assertEqual(provider, 'mistral')
        
        provider = self.llm_wrapper._get_provider_for_model('llama-13b')
        self.assertEqual(provider, 'local')

if __name__ == '__main__':
    unittest.main() 