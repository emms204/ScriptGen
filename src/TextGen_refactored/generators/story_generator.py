"""
Story generator implementation.

This module provides the core story generation functionality.
"""
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import re

from ..models.core import Title, Character, Characters, Place, Scene, Scenes, Story
from ..providers import LanguageModelProvider, FilterProvider
from ..prompts import render_template
from ..config.constants import MARKERS, SAMPLE_LENGTH, MAX_PARAGRAPH_LENGTH


@dataclass
class GenerationHistory:
    """History of generation actions."""
    
    class Action:
        """Type of generation action."""
        NEW = 1
        CONTINUE = 2
        REWRITE = 3
    
    items: List[Any] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    position: int = field(default=0)
    
    def add(self, item: Any, action: int) -> None:
        """Add an item to the history.
        
        Args:
            item: Item to add
            action: Action type
        """
        if self.position < len(self.items):
            # We're not at the end of the history, so truncate
            self.items = self.items[:self.position + 1]
            self.actions = self.actions[:self.position + 1]
        
        self.items.append(item)
        self.actions.append(action)
        self.position = len(self.items) - 1
    
    def previous(self) -> Optional[Tuple[Any, int]]:
        """Get the previous item and action in the history.
        
        Returns:
            Tuple of (item, action) or None if at the beginning
        """
        if self.position > 0:
            self.position -= 1
            return self.items[self.position], self.actions[self.position]
        return None
    
    def next(self) -> Optional[Tuple[Any, int]]:
        """Get the next item and action in the history.
        
        Returns:
            Tuple of (item, action) or None if at the end
        """
        if self.position < len(self.items) - 1:
            self.position += 1
            return self.items[self.position], self.actions[self.position]
        return None
    
    def current(self) -> Optional[Tuple[Any, int]]:
        """Get the current item and action in the history.
        
        Returns:
            Tuple of (item, action) or None if history is empty
        """
        if self.items:
            return self.items[self.position], self.actions[self.position]
        return None


class StoryGenerator:
    """Generate a story from a provided storyline.
    
    This class orchestrates the generation of different story elements
    (title, characters, scenes, places, dialogs) in a hierarchical manner.
    """
    
    # Names of the hierarchical levels
    LEVELS = ['storyline', 'title', 'characters', 'scenes', 'places', 'dialogs']
    
    def __init__(
        self,
        storyline: str,
        language_provider: LanguageModelProvider,
        filter_provider: Optional[FilterProvider] = None,
        max_paragraph_length: int = MAX_PARAGRAPH_LENGTH['DEFAULT'],
        max_paragraph_length_characters: int = MAX_PARAGRAPH_LENGTH['CHARACTERS'],
        max_paragraph_length_scenes: int = MAX_PARAGRAPH_LENGTH['SCENES'],
        num_samples: int = 1,
        seed: Optional[int] = None
    ):
        """Initialize the story generator.
        
        Args:
            storyline: The storyline to generate from
            language_provider: Provider for language model generations
            filter_provider: Optional provider for content filtering
            max_paragraph_length: Maximum paragraph length for general text
            max_paragraph_length_characters: Maximum paragraph length for characters
            max_paragraph_length_scenes: Maximum paragraph length for scenes
            num_samples: Number of samples to generate
            seed: Random seed for reproducibility
        """
        self._storyline = storyline.strip()
        self._language_provider = language_provider
        self._filter_provider = filter_provider
        self._max_paragraph_length = max_paragraph_length
        self._max_paragraph_length_characters = max_paragraph_length_characters
        self._max_paragraph_length_scenes = max_paragraph_length_scenes
        self._num_samples = num_samples
        self._seed = seed
        
        # Initialize story elements
        self._title = None
        self._characters = None
        self._scenes = None
        self._places = {}
        self._dialogs = []
        
        # Initialize histories
        self._title_history = GenerationHistory()
        self._characters_history = GenerationHistory()
        self._scenes_history = GenerationHistory()
        self._places_history = GenerationHistory()
        self._dialogs_history = GenerationHistory()
    
    @property
    def storyline(self) -> str:
        """Get the storyline."""
        return self._storyline
    
    @property
    def title(self) -> Optional[Title]:
        """Get the title."""
        return self._title
    
    @property
    def characters(self) -> Optional[Characters]:
        """Get the characters."""
        return self._characters
    
    @property
    def scenes(self) -> Optional[Scenes]:
        """Get the scenes."""
        return self._scenes
    
    @property
    def places(self) -> Dict[str, Place]:
        """Get the places."""
        return self._places
    
    @property
    def dialogs(self) -> List[str]:
        """Get the dialogs."""
        return self._dialogs
    
    def _filter_text(self, text: str) -> str:
        """Filter text based on content policy.
        
        Args:
            text: Text to filter
            
        Returns:
            Filtered text
        """
        if self._filter_provider and not self._filter_provider.validate(text):
            # If the text doesn't pass validation, generate a safer alternative
            safer_text = self._generate_text(
                render_template('regenerate', text=text),
                sample_length=len(text) * 2
            )
            return safer_text
        return text
    
    def _generate_text(
        self,
        prompt: str,
        sample_length: Optional[int] = None,
        temperature: float = 0.7,
        seed: Optional[int] = None,
        max_paragraph_length: Optional[int] = None
    ) -> str:
        """Generate text using the language provider.
        
        Args:
            prompt: Prompt for generation
            sample_length: Maximum sample length
            temperature: Temperature for generation
            seed: Random seed
            max_paragraph_length: Maximum paragraph length
            
        Returns:
            Generated text
        """
        if sample_length is None:
            sample_length = self._language_provider.default_sample_length
            
        if seed is None:
            seed = self._seed
        
        if max_paragraph_length is None:
            max_paragraph_length = self._max_paragraph_length
            
        # Generate text
        responses = self._language_provider.sample(
            prompt=prompt,
            sample_length=sample_length,
            temperature=temperature,
            seed=seed,
            num_samples=self._num_samples
        )
        
        if not responses:
            return ""
            
        # Use the first response
        text = responses[0].text
        
        # Clean up the text
        if text.endswith(MARKERS['END']):
            text = text[:text.rfind(MARKERS['END'])].strip()
            
        # Apply content filtering if necessary
        return self._filter_text(text)
        
    def generate_title(self) -> Title:
        """Generate a title from the storyline.
        
        Returns:
            Generated title
        """
        # Generate title
        prompt = render_template('title', logline=self.storyline)
        title_text = self._generate_text(
            prompt=prompt,
            sample_length=SAMPLE_LENGTH['TITLE']
        )
        
        # Create Title object
        title = Title(title=title_text)
        
        # Update state
        self._title = title
        self._title_history.add(title, GenerationHistory.Action.NEW)
        
        return title
    
    def generate_characters(
        self,
        predefined_characters: Optional[str] = None
    ) -> Characters:
        """Generate characters from the storyline and title.
        
        Args:
            predefined_characters: Optional predefined character descriptions
            
        Returns:
            Generated characters
        """
        if self._title is None:
            self.generate_title()
            
        # Generate characters
        if predefined_characters:
            prompt = render_template(
                'characters_with_initial_data',
                logline=self.storyline,
                title=self._title.title,
                initial_character_data=predefined_characters
            )
        else:
            prompt = render_template(
                'characters',
                logline=self.storyline,
                title=self._title.title
            )
            
        characters_text = self._generate_text(
            prompt=prompt,
            max_paragraph_length=self._max_paragraph_length_characters
        )
        
        # Create Characters object
        characters = Characters.from_string(characters_text)
        
        # Update state
        self._characters = characters
        self._characters_history.add(characters, GenerationHistory.Action.NEW)
        
        return characters
    
    def generate_scenes(self, example_scenes: str) -> Scenes:
        """Generate scenes from the storyline, title, and characters.
        
        Args:
            example_scenes: Example scenes to guide generation
            
        Returns:
            Generated scenes
        """
        if self._title is None:
            self.generate_title()
            
        if self._characters is None:
            self.generate_characters()
            
        # Format character descriptions as a string
        character_str = ""
        for name, desc in self._characters.character_descriptions.items():
            character_str += f"{name} - {desc}\n\n"
            
        # Generate scenes
        prompt = render_template(
            'scenes',
            logline=self.storyline,
            title=self._title,
            character_descriptions=character_str,
            example=example_scenes
        )
        
        scenes_text = self._generate_text(
            prompt=prompt,
            max_paragraph_length=self._max_paragraph_length_scenes
        )
        
        # Create Scenes object
        scenes = Scenes.from_string(scenes_text)
        
        # Update state
        self._scenes = scenes
        self._scenes_history.add(scenes, GenerationHistory.Action.NEW)
        
        return scenes
    
    def generate_place(self, place_name: str) -> Place:
        """Generate a description for a place.
        
        Args:
            place_name: Name of the place to describe
            
        Returns:
            Generated place
        """
        # Generate place description
        prompt = render_template(
            'place',
            logline=self.storyline,
            place_name=place_name
        )
        
        place_text = self._generate_text(
            prompt=prompt,
            sample_length=SAMPLE_LENGTH['PLACE']
        )
        
        # Create Place object
        place = Place(name=place_name, description=place_text)
        
        # Update state
        self._places[place_name] = place
        self._places_history.add(self._places.copy(), GenerationHistory.Action.NEW)
        
        return place
    
    def generate_places(self) -> Dict[str, Place]:
        """Generate descriptions for all places in the scenes.
        
        Returns:
            Dictionary of place names to Place objects
        """
        if self._scenes is None:
            raise ValueError("Cannot generate places without scenes")
            
        # Generate places for each unique place in the scenes
        place_names = set(scene.place for scene in self._scenes.scenes)
        
        for place_name in place_names:
            if place_name not in self._places:
                self.generate_place(place_name)
                
        return self._places
    
    def generate_dialog(self, scene_index: int) -> str:
        """Generate dialog for a specific scene.
        
        Args:
            scene_index: Index of the scene to generate dialog for
            
        Returns:
            Generated dialog
        """
        if self._scenes is None or scene_index >= len(self._scenes.scenes):
            raise ValueError(f"Invalid scene index: {scene_index}")
            
        if not self._places:
            self.generate_places()
            
        # Get the scene and place
        scene = self._scenes.scenes[scene_index]
        place = self._places.get(scene.place)
        
        if not place:
            raise ValueError(f"Place not found: {scene.place}")
            
        # Format character descriptions as a string
        character_str = ""
        for name, desc in self._characters.character_descriptions.items():
            character_str += f"{name} - {desc}\n\n"
            
        # Check if we should continue from previous dialog
        is_continuation = (
            scene_index > 0 and 
            scene_index <= len(self._dialogs) and 
            bool(self._dialogs)
        )
        
        if is_continuation:
            # Continue from previous dialog
            prompt = render_template(
                'dialog_sequence',
                logline=self.storyline,
                character_descriptions=character_str,
                scene=scene.to_string(),
                place_description=place.description,
                previous_dialog=self._dialogs[scene_index - 1]
            )
        else:
            # Generate new dialog
            prompt = render_template(
                'dialog',
                logline=self.storyline,
                character_descriptions=character_str,
                scene=scene.to_string(),
                place_description=place.description
            )
            
        dialog = self._generate_text(prompt=prompt)
        
        # Update state
        if scene_index < len(self._dialogs):
            self._dialogs[scene_index] = dialog
        else:
            self._dialogs.append(dialog)
            
        self._dialogs_history.add(self._dialogs.copy(), GenerationHistory.Action.NEW)
        
        return dialog
    
    def generate_all_dialogs(self) -> List[str]:
        """Generate dialogs for all scenes in sequence.
        
        Returns:
            List of generated dialogs
        """
        if self._scenes is None:
            raise ValueError("Cannot generate dialogs without scenes")
            
        if not self._places:
            self.generate_places()
            
        # Clear existing dialogs
        self._dialogs = []
        
        # Generate dialog for each scene
        for i in range(len(self._scenes.scenes)):
            self.generate_dialog(i)
            
        return self._dialogs
    
    def rewrite_title(self, text: Optional[str] = None) -> Title:
        """Rewrite the title.
        
        Args:
            text: Optional new title text
            
        Returns:
            Rewritten title
        """
        if text:
            # Use provided text
            title = Title(title=text)
        else:
            # Generate new title
            prompt = render_template('title', logline=self.storyline)
            title_text = self._generate_text(
                prompt=prompt,
                sample_length=SAMPLE_LENGTH['TITLE']
            )
            title = Title(title=title_text)
            
        # Update state
        self._title = title
        self._title_history.add(title, GenerationHistory.Action.REWRITE)
        
        return title
    
    def rewrite_characters(self, text: Optional[str] = None) -> Characters:
        """Rewrite the characters.
        
        Args:
            text: Optional new characters text
            
        Returns:
            Rewritten characters
        """
        if text:
            # Use provided text
            characters = Characters.from_string(text)
        else:
            # Generate new characters
            return self.generate_characters()
            
        # Update state
        self._characters = characters
        self._characters_history.add(characters, GenerationHistory.Action.REWRITE)
        
        return characters
    
    def rewrite_scene(
        self,
        index: int,
        text: Optional[str] = None,
        place: Optional[str] = None,
        plot_element: Optional[str] = None,
        beat: Optional[str] = None
    ) -> Scenes:
        """Rewrite a specific scene.
        
        Args:
            index: Index of the scene to rewrite
            text: Optional new scene text
            place: Optional new place
            plot_element: Optional new plot element
            beat: Optional new beat
            
        Returns:
            Updated scenes
        """
        if self._scenes is None or index >= len(self._scenes.scenes):
            raise ValueError(f"Invalid scene index: {index}")
            
        scenes_copy = Scenes(scenes=self._scenes.scenes.copy())
        
        if text:
            # Parse the text as a scene
            scene = Scene.from_string(text)
            scenes_copy.scenes[index] = scene
        else:
            # Update individual fields
            scene = scenes_copy.scenes[index]
            
            if place:
                scene.place = place
                
            if plot_element:
                scene.plot_element = plot_element
                
            if beat:
                scene.beat = beat
                
        # Update state
        self._scenes = scenes_copy
        self._scenes_history.add(scenes_copy, GenerationHistory.Action.REWRITE)
        
        return scenes_copy
    
    def rewrite_place(self, place_name: str, text: Optional[str] = None) -> Place:
        """Rewrite a place description.
        
        Args:
            place_name: Name of the place to rewrite
            text: Optional new place description
            
        Returns:
            Updated place
        """
        if place_name not in self._places:
            raise ValueError(f"Place not found: {place_name}")
            
        if text:
            # Use provided text
            place = Place(name=place_name, description=text)
        else:
            # Generate new place description
            return self.generate_place(place_name)
            
        # Update state
        self._places[place_name] = place
        self._places_history.add(self._places.copy(), GenerationHistory.Action.REWRITE)
        
        return place
    
    def rewrite_dialog(self, index: int, text: Optional[str] = None) -> str:
        """Rewrite a dialog.
        
        Args:
            index: Index of the dialog to rewrite
            text: Optional new dialog text
            
        Returns:
            Updated dialog
        """
        if index >= len(self._dialogs):
            raise ValueError(f"Invalid dialog index: {index}")
            
        if text:
            # Use provided text
            dialog = text
        else:
            # Regenerate the dialog
            return self.generate_dialog(index)
            
        # Update state
        self._dialogs[index] = dialog
        self._dialogs_history.add(self._dialogs.copy(), GenerationHistory.Action.REWRITE)
        
        return dialog
    
    def get_story(self) -> Story:
        """Get the complete story.
        
        Returns:
            Complete story
        """
        if self._title is None:
            raise ValueError("Title not generated")
            
        if self._characters is None:
            raise ValueError("Characters not generated")
            
        if self._scenes is None:
            raise ValueError("Scenes not generated")
            
        if not self._places:
            raise ValueError("Places not generated")
            
        if not self._dialogs or len(self._dialogs) < len(self._scenes.scenes):
            raise ValueError("Dialogs not fully generated")
            
        # Create the story
        return Story(
            storyline=self._storyline,
            title=self._title.title,
            character_descriptions=self._characters.character_descriptions,
            place_descriptions=self._places,
            scenes=self._scenes,
            dialogs=self._dialogs
        )
    
    def generate_complete_story(self, example_scenes: str) -> Story:
        """Generate a complete story from the storyline.
        
        Args:
            example_scenes: Example scenes to guide generation
            
        Returns:
            Complete story
        """
        # Generate all story elements
        self.generate_title()
        self.generate_characters()
        self.generate_scenes(example_scenes)
        self.generate_places()
        self.generate_all_dialogs()
        
        # Return the complete story
        return self.get_story() 