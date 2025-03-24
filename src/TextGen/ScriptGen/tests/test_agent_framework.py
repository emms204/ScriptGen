import unittest
from unittest.mock import patch, MagicMock
import json

from ..agents.agent_framework import (
    CharacterAgent,
    DirectorAgent,
    DramaturgeAgent,
    run_agents,
    batch_character_agent_call
)

class TestAgentFramework(unittest.TestCase):
    """Test cases for the Agent Framework module"""
    
    def setUp(self):
        """Set up test environment"""
        # Create sample character bibles
        self.character1 = {
            'id': 1,
            'type': 'character',
            'content': {
                'name': 'Detective Smith',
                'backstory': 'A veteran detective with amnesia',
                'goals': 'Solve the murder case',
                'fears': 'Discovering he committed the crime',
                'traits': ['determined', 'confused', 'intelligent']
            }
        }
        
        self.character2 = {
            'id': 2,
            'type': 'character',
            'content': {
                'name': 'Dr. Wilson',
                'backstory': 'A forensic psychologist helping the detective',
                'goals': 'Uncover the truth',
                'fears': 'Being misled by the detective',
                'traits': ['analytical', 'cautious', 'empathetic']
            }
        }
        
        # Create sample plot bible
        self.plot_bible = {
            'id': 3,
            'type': 'plot',
            'content': {
                'scene_id': 'act1_scene2',
                'act': 1,
                'description': 'Detective Smith examines crime scene with Dr. Wilson',
                'prior_state': 'Detective has just discovered the victim was someone from his past',
                'next_state': 'Detective realizes some evidence implicates him'
            }
        }
        
        # Create sample scene context
        self.scene_context = {
            'location': 'Crime scene - Abandoned warehouse',
            'time': 'Night',
            'description': 'Detective Smith and Dr. Wilson examine the crime scene. The detective experiences flashes of memory.',
            'objectives': 'Reveal the victim\'s identity and establish detective\'s connection to them'
        }
    
    @patch('TextGen.ScriptGen.agents.agent_framework.LLMWrapper')
    def test_character_agent(self, mock_llm_wrapper):
        """Test the character agent dialogue generation"""
        # Set up mock
        mock_instance = MagicMock()
        mock_llm_wrapper.return_value = mock_instance
        
        # Mock response from LLM
        mock_instance.call_llm.return_value = {
            'text': '1. (confused, examining a photo) Is that... do I know this person?\n'
                    '2. (frustrated) I can\'t remember. Why can\'t I remember?\n'
                    '3. (determined) There\'s something familiar about the wound pattern.'
        }
        
        # Create agent and generate dialogue
        agent = CharacterAgent()
        result = agent.generate_dialogue(self.character1, json.dumps(self.scene_context))
        
        # Verify results
        self.assertEqual(result['character'], 'Detective Smith')
        self.assertEqual(len(result['dialogue_lines']), 3)
        self.assertEqual(result['dialogue_lines'][0]['action'], '(confused, examining a photo)')
        self.assertEqual(result['dialogue_lines'][0]['text'], 'Is that... do I know this person?')
    
    @patch('TextGen.ScriptGen.agents.agent_framework.LLMWrapper')
    def test_director_agent(self, mock_llm_wrapper):
        """Test the director agent scene integration"""
        # Set up mock
        mock_instance = MagicMock()
        mock_llm_wrapper.return_value = mock_instance
        
        # Mock response from LLM
        mock_instance.call_llm.return_value = {
            'text': 'INT. ABANDONED WAREHOUSE - NIGHT\n\n'
                    'Pools of water reflect the flashing police lights. '
                    'DETECTIVE SMITH kneels beside the body, examining it with gloved hands.'
        }
        
        # Create sample character outputs
        character_outputs = [
            {
                'character': 'Detective Smith',
                'dialogue_lines': [
                    {'character': 'Detective Smith', 'action': '(examining body)', 'text': 'This wound pattern...'}
                ]
            },
            {
                'character': 'Dr. Wilson',
                'dialogue_lines': [
                    {'character': 'Dr. Wilson', 'action': '(concerned)', 'text': 'You recognize something?'}
                ]
            }
        ]
        
        # Create agent and integrate scene
        agent = DirectorAgent()
        result = agent.integrate_scene(character_outputs, self.plot_bible, self.scene_context)
        
        # Verify results
        self.assertIn('INT. ABANDONED WAREHOUSE - NIGHT', result['integrated_scene'])
        self.assertIn('DETECTIVE SMITH', result['integrated_scene'])
    
    @patch('TextGen.ScriptGen.agents.agent_framework.LLMWrapper')
    def test_dramaturge_agent(self, mock_llm_wrapper):
        """Test the dramaturge agent scene critique"""
        # Set up mock
        mock_instance = MagicMock()
        mock_llm_wrapper.return_value = mock_instance
        
        # Mock response from LLM
        mock_instance.call_llm.return_value = {
            'text': 'OVERALL ASSESSMENT:\n'
                    'The scene effectively establishes tension but could benefit from more subtext.'
        }
        
        # Create sample scene
        scene = 'INT. ABANDONED WAREHOUSE - NIGHT\n\n' \
                'Detective Smith examines the body while Dr. Wilson watches.\n\n' \
                'DETECTIVE SMITH\n' \
                'I know this person.\n\n' \
                'DR. WILSON\n' \
                'Are you sure?'
        
        # Create agent and critique scene
        agent = DramaturgeAgent()
        result = agent.critique_scene(scene, self.plot_bible)
        
        # Verify results
        self.assertIn('OVERALL ASSESSMENT', result['critique'])
        self.assertIn('tension', result['critique'])
    
    @patch('TextGen.ScriptGen.agents.agent_framework.CharacterAgent')
    @patch('TextGen.ScriptGen.agents.agent_framework.DirectorAgent')
    @patch('TextGen.ScriptGen.agents.agent_framework.DramaturgeAgent')
    def test_run_agents(self, mock_dramaturge, mock_director, mock_character):
        """Test the run_agents orchestration function"""
        # Set up mocks
        mock_char_instance = MagicMock()
        mock_dir_instance = MagicMock()
        mock_dram_instance = MagicMock()
        
        mock_character.return_value = mock_char_instance
        mock_director.return_value = mock_dir_instance
        mock_dramaturge.return_value = mock_dram_instance
        
        # Set up returns
        mock_char_instance.generate_dialogue.return_value = {
            'character': 'Detective Smith',
            'dialogue_lines': [{'character': 'Detective Smith', 'action': '', 'text': 'Test line'}]
        }
        
        mock_dir_instance.integrate_scene.return_value = {
            'integrated_scene': 'INT. WAREHOUSE - NIGHT\n\nTest scene.'
        }
        
        mock_dram_instance.critique_scene.return_value = {
            'critique': 'Test critique'
        }
        
        # Call run_agents
        result = run_agents(
            'act1_scene2',
            [self.character1, self.character2],
            self.plot_bible,
            self.scene_context
        )
        
        # Verify results
        self.assertEqual(result['scene_id'], 'act1_scene2')
        self.assertEqual(result['outline'], self.scene_context)
        self.assertIn('character_dialogue', result)
        self.assertEqual(result['integrated_scene'], 'INT. WAREHOUSE - NIGHT\n\nTest scene.')
        self.assertEqual(result['critique'], 'Test critique')
        
        # Verify each agent was called with appropriate args
        self.assertEqual(mock_char_instance.generate_dialogue.call_count, 2)
        mock_dir_instance.integrate_scene.assert_called_once()
        mock_dram_instance.critique_scene.assert_called_once()
    
    @patch('TextGen.ScriptGen.agents.agent_framework.LLMWrapper')
    def test_batch_character_agent_call(self, mock_llm_wrapper):
        """Test batching character agent calls"""
        # Set up mock
        mock_instance = MagicMock()
        mock_llm_wrapper.return_value = mock_instance
        
        # Mock response from LLM
        mock_instance.call_llm.return_value = {
            'text': '(confused) Who was this person?\n'
                    '(determined) I need to remember.\n'
                    '===CHARACTER END===\n'
                    '(analytical) The wound pattern suggests...\n'
                    '(concerned) Detective, are you alright?'
        }
        
        # Call batch function
        result = batch_character_agent_call(
            [self.character1, self.character2],
            json.dumps(self.scene_context)
        )
        
        # Verify results
        self.assertEqual(len(result['character_outputs']), 2)
        self.assertEqual(result['character_outputs'][0]['character'], 'Detective Smith')
        self.assertEqual(result['character_outputs'][1]['character'], 'Dr. Wilson')
        
        # Check first character's lines
        self.assertEqual(len(result['character_outputs'][0]['dialogue_lines']), 2)
        self.assertEqual(result['character_outputs'][0]['dialogue_lines'][0]['action'], '(confused)')
        self.assertEqual(result['character_outputs'][0]['dialogue_lines'][0]['text'], 'Who was this person?')
        
        # Check second character's lines
        self.assertEqual(len(result['character_outputs'][1]['dialogue_lines']), 2)
        self.assertEqual(result['character_outputs'][1]['dialogue_lines'][0]['action'], '(analytical)')
        self.assertEqual(result['character_outputs'][1]['dialogue_lines'][1]['text'], 'Detective, are you alright?')


if __name__ == '__main__':
    unittest.main() 