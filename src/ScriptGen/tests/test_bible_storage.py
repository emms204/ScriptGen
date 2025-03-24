import unittest
import os
import json
from ..bibles.bible_storage import BibleStorage

class TestBibleStorage(unittest.TestCase):
    """Test cases for the BibleStorage module"""
    
    def setUp(self):
        """Set up test environment"""
        # Use in-memory database for testing
        self.bible_storage = BibleStorage(db_path=':memory:')
        
        # Sample data for tests
        self.char_data = {
            "name": "Test Character",
            "backstory": "A character for testing",
            "goals": "To test the system",
            "fears": "Bugs and errors",
            "traits": ["brave", "resourceful"]
        }
        
        self.plot_data = {
            "scene_id": "test_scene",
            "description": "A test scene",
            "prior_state": "Before the test",
            "next_state": "After the test"
        }
    
    def test_create_entry(self):
        """Test creating new Bible entries"""
        # Test creating character bible
        char_entry = self.bible_storage.create_entry("character", self.char_data)
        self.assertIsNotNone(char_entry)
        self.assertEqual(char_entry['type'], 'character')
        self.assertEqual(char_entry['content']['name'], 'Test Character')
        
        # Test creating plot bible
        plot_entry = self.bible_storage.create_entry("plot", self.plot_data)
        self.assertIsNotNone(plot_entry)
        self.assertEqual(plot_entry['type'], 'plot')
        self.assertEqual(plot_entry['content']['scene_id'], 'test_scene')
    
    def test_create_and_get_bible(self):
        """Test creating and retrieving a bible entry"""
        # Create character bible
        character_id = self.bible_storage.create_bible('character', self.char_data)
        
        # Retrieve character bible
        character = self.bible_storage.get_bible(character_id)
        
        # Verify data
        self.assertEqual(character['content']['name'], 'Test Character')
        self.assertEqual(character['type'], 'character')
    
    def test_update_bible(self):
        """Test updating a bible entry"""
        # Create character bible
        character_id = self.bible_storage.create_bible('character', self.char_data)
        
        # Update name
        updated_data = self.char_data.copy()
        updated_data['name'] = 'Updated Test Character'
        
        self.bible_storage.update_bible(character_id, updated_data)
        
        # Retrieve updated character
        character = self.bible_storage.get_bible(character_id)
        
        # Verify update
        self.assertEqual(character['content']['name'], 'Updated Test Character')
        
        # Check version history
        self.assertEqual(len(json.loads(character['version_history'])), 1)
    
    def test_get_all_bibles(self):
        """Test retrieving all bibles of a specific type"""
        # Create multiple bibles
        self.bible_storage.create_bible('character', self.char_data)
        self.bible_storage.create_bible('plot', self.plot_data)
        
        # Get all characters
        characters = self.bible_storage.get_all_bibles('character')
        
        # Verify count
        self.assertEqual(len(characters), 1)
        
        # Verify name
        self.assertEqual(characters[0]['content']['name'], 'Test Character')
    
    def test_delete_bible(self):
        """Test deleting a bible entry"""
        # Create character bible
        character_id = self.bible_storage.create_bible('character', self.char_data)
        
        # Delete it
        self.bible_storage.delete_bible(character_id)
        
        # Verify it's gone
        with self.assertRaises(ValueError):
            self.bible_storage.get_bible(character_id)
    
    def test_revert_to_version(self):
        """Test reverting to a previous version"""
        # Create character bible
        character_id = self.bible_storage.create_bible('character', self.char_data)
        
        # Make first update
        update1 = self.char_data.copy()
        update1['name'] = 'Updated Test Character'
        self.bible_storage.update_bible(character_id, update1)
        
        # Make second update
        update2 = update1.copy()
        update2['backstory'] = 'A bitter ex-musician seeking revenge.'
        self.bible_storage.update_bible(character_id, update2)
        
        # Get current version to check history count
        current = self.bible_storage.get_bible(character_id)
        version_history = json.loads(current['version_history'])
        self.assertEqual(len(version_history), 2)
        
        # Revert to first version
        self.bible_storage.revert_to_version(character_id, 0)
        
        # Get reverted version
        reverted = self.bible_storage.get_bible(character_id)
        
        # Verify we're back to the first update
        self.assertEqual(reverted['content']['name'], 'Test Character')
        self.assertEqual(reverted['content']['backstory'], self.char_data['backstory'])

if __name__ == '__main__':
    unittest.main() 