"""
Enhanced story generator with additional features like toxicity filtering and storyboard generation.

This module extends the basic story generator with advanced features.
"""
from typing import Dict, List, Optional, Any, Set, Tuple
import re

from ..models.core import Title, Characters, Scene, Scenes, Place, Story
from ..providers import LanguageModelProvider, FilterProvider, ProviderFactory
from ..prompts import render_template
from ..utils import parse_toxicity_rating, render_story
from .story_generator import StoryGenerator


class EnhancedStoryGenerator(StoryGenerator):
    """Enhanced story generator with additional features.
    
    This class extends the basic StoryGenerator with features like
    toxicity filtering, storyboard generation, and more.
    """
    
    def __init__(
        self,
        storyline: str,
        language_provider: LanguageModelProvider,
        filter_provider: Optional[FilterProvider] = None,
        analyze_toxicity: bool = False,
        toxicity_provider: Optional[LanguageModelProvider] = None,
        toxicity_threshold: float = 0.55,
        dialog_sequence: bool = True,
        **kwargs
    ):
        """Initialize the enhanced story generator.
        
        Args:
            storyline: The storyline to generate from
            language_provider: Provider for language model generations
            filter_provider: Optional provider for content filtering
            analyze_toxicity: Whether to analyze and filter toxicity
            toxicity_provider: Provider for toxicity analysis (uses OpenAI by default)
            toxicity_threshold: Threshold for toxicity filtering
            dialog_sequence: Whether to generate dialog in sequence
            **kwargs: Additional arguments for the base StoryGenerator
        """
        super().__init__(
            storyline=storyline,
            language_provider=language_provider,
            filter_provider=filter_provider,
            **kwargs
        )
        
        self._analyze_toxicity = analyze_toxicity
        self._toxicity_threshold = toxicity_threshold
        self._dialog_sequence = dialog_sequence
        
        # Initialize toxicity provider if needed
        if analyze_toxicity and not toxicity_provider:
            self._toxicity_provider = ProviderFactory.create_provider(
                provider_type='openai',
                model_name='gpt-4o-mini'
            )
        else:
            self._toxicity_provider = toxicity_provider
    
    def _check_toxicity(self, text: str) -> List[float]:
        """Check the toxicity level of text.
        
        Args:
            text: Text to check
            
        Returns:
            List of toxicity ratings (violence, hate speech, sexual content, self-harm, harassment)
        """
        if not self._analyze_toxicity or not self._toxicity_provider:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
            
        prompt = render_template('toxicity', text=text)
        
        responses = self._toxicity_provider.sample(
            prompt=prompt,
            sample_length=15,
            temperature=0.0
        )
        
        if not responses:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
            
        return parse_toxicity_rating(responses[0].text)
    
    def _filter_toxicity(self, text: str, temperature: float = 0.7) -> str:
        """Filter toxic content from text.
        
        Args:
            text: Text to filter
            temperature: Temperature for regeneration
            
        Returns:
            Filtered text
        """
        if not self._analyze_toxicity:
            return text
            
        toxicity_ratings = self._check_toxicity(text)
        
        if max(toxicity_ratings) > self._toxicity_threshold:
            # Regenerate the text with lower toxicity
            prompt = render_template('regenerate', text=text)
            
            responses = self._toxicity_provider.sample(
                prompt=prompt,
                sample_length=int(len(text.split()) * 1.5),
                temperature=temperature
            )
            
            if responses:
                return responses[0].text
                
        return text
    
    def _generate_text(
        self,
        prompt: str,
        sample_length: Optional[int] = None,
        temperature: float = 0.7,
        seed: Optional[int] = None,
        max_paragraph_length: Optional[int] = None
    ) -> str:
        """Generate text with toxicity filtering.
        
        Args:
            prompt: Prompt for generation
            sample_length: Maximum sample length
            temperature: Temperature for generation
            seed: Random seed
            max_paragraph_length: Maximum paragraph length
            
        Returns:
            Generated text
        """
        # Generate text using the parent method
        text = super()._generate_text(
            prompt=prompt,
            sample_length=sample_length,
            temperature=temperature,
            seed=seed,
            max_paragraph_length=max_paragraph_length
        )
        
        # Apply toxicity filtering
        return self._filter_toxicity(text, temperature)
    
    def generate_dialog(self, scene_index: int) -> str:
        """Generate dialog for a scene with sequence awareness.
        
        Args:
            scene_index: Index of the scene
            
        Returns:
            Generated dialog
        """
        if not self._dialog_sequence or scene_index == 0:
            # For the first scene or if sequence mode is off, use parent method
            return super().generate_dialog(scene_index)
            
        # For subsequent scenes in sequence mode, use the previous dialog for context
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
            
        # Get the previous dialog
        if scene_index > 0 and scene_index <= len(self._dialogs) and self._dialogs:
            previous_dialog = self._dialogs[scene_index - 1]
            
            # Generate dialog that continues from the previous scene
            prompt = render_template(
                'dialog_sequence',
                logline=self.storyline,
                character_descriptions=character_str,
                scene=scene.to_string(),
                place_description=place.description,
                previous_dialog=previous_dialog
            )
        else:
            # Fall back to regular dialog generation
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
            
        self._dialogs_history.add(self._dialogs.copy(), self._dialogs_history.Action.NEW)
        
        return dialog
    
    def parse_storyboard_elements(self, storyboard_text: str) -> Tuple[List[str], List[str], List[str]]:
        """Parse storyboard elements from text.
        
        Args:
            storyboard_text: Storyboard text to parse
            
        Returns:
            Tuple of (visuals, text, characters)
        """
        # Initialize empty lists for each element
        visuals = []
        text_elements = []
        characters = []
        
        # Parse Visuals
        visual_pattern = re.compile(r'\*\*Visuals\*\*:(.*?)(?=\n-|\n\n|\Z)', re.DOTALL)
        visual_matches = visual_pattern.findall(storyboard_text)
        visuals = [v.strip() for v in visual_matches]
        
        # Parse Text
        text_pattern = re.compile(r'\*\*Text\*\*:(.*?)(?=\n-|\n\n|\Z)', re.DOTALL)
        text_matches = text_pattern.findall(storyboard_text)
        text_elements = [t.strip().strip('"') for t in text_matches]
        
        # Parse Characters
        character_pattern = re.compile(r'\*\*Characters\*\*:(.*?)(?=\n\n|\Z)', re.DOTALL)
        character_matches = character_pattern.findall(storyboard_text)
        
        for match in character_matches:
            chars = [c.strip() for c in match.split(',') if c.strip() != "None" and c.strip() != "None visible"]
            characters.extend(chars)
        
        # Remove duplicates from characters list
        characters = list(dict.fromkeys(characters))
        
        return visuals, text_elements, characters
    
    def generate_storyboard(self, scene_index: int) -> Tuple[List[str], List[str], List[str]]:
        """Generate a storyboard for a scene.
        
        Args:
            scene_index: Index of the scene
            
        Returns:
            Tuple of (visuals, text, characters)
        """
        if self._scenes is None or scene_index >= len(self._scenes.scenes):
            raise ValueError(f"Invalid scene index: {scene_index}")
            
        if not self._dialogs or scene_index >= len(self._dialogs):
            raise ValueError(f"Dialog not generated for scene {scene_index}")
            
        # Get the scene, place, and dialog
        scene = self._scenes.scenes[scene_index]
        place = self._places.get(scene.place)
        dialog = self._dialogs[scene_index]
        
        if not place:
            raise ValueError(f"Place not found: {scene.place}")
            
        # Create a script excerpt for the scene
        script_excerpt = f"""
        Scene: {scene_index + 1}
        Place: {scene.place}
        Plot element: {scene.plot_element}
        Beat: {scene.beat}
        
        Place Description:
        {place.description}
        
        Dialog:
        {dialog}
        """
        
        # Generate the storyboard
        prompt = render_template('storyboards', script_text=script_excerpt)
        
        storyboard_text = self._generate_text(
            prompt=prompt,
            sample_length=4096
        )
        
        # Parse the storyboard elements
        return self.parse_storyboard_elements(storyboard_text)
    
    def generate_all_storyboards(self) -> List[Tuple[List[str], List[str], List[str]]]:
        """Generate storyboards for all scenes.
        
        Returns:
            List of storyboard element tuples (visuals, text, characters)
        """
        if self._scenes is None:
            raise ValueError("Cannot generate storyboards without scenes")
            
        if not self._dialogs or len(self._dialogs) < len(self._scenes.scenes):
            raise ValueError("Dialogs not fully generated")
            
        storyboards = []
        
        for i in range(len(self._scenes.scenes)):
            storyboards.append(self.generate_storyboard(i))
            
        return storyboards
    
    def get_storyboard_mapping(self) -> List[Dict[str, str]]:
        """Get a mapping of storyboard text to visuals.
        
        Returns:
            List of dictionaries mapping text to visuals
        """
        storyboards = self.generate_all_storyboards()
        
        mapping = []
        
        for visuals, texts, _ in storyboards:
            scene_mapping = {}
            
            for text, visual in zip(texts, visuals):
                scene_mapping[text] = visual
                
            mapping.append(scene_mapping)
            
        return mapping 