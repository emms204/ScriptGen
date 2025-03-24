"""
Core data models for DramaTron.

This module defines the foundational data structures used throughout the DramaTron system.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, ClassVar, Union
import re
from ..config.constants import ELEMENTS, MARKERS


@dataclass
class Title:
    """Title of a story or script."""
    title: str
    
    @classmethod
    def from_string(cls, text: str) -> 'Title':
        """Parse a title from a string.
        
        Args:
            text: String containing the title, typically prefixed with "Title: "
            
        Returns:
            A Title instance
        """
        if text.startswith(ELEMENTS['TITLE']):
            title_text = text[len(ELEMENTS['TITLE']):].strip()
        else:
            title_text = text.strip()
        
        return cls(title=title_text)
    
    def to_string(self) -> str:
        """Convert the title to a formatted string.
        
        Returns:
            Formatted title string
        """
        return f"{ELEMENTS['TITLE']}{self.title}"


@dataclass
class Character:
    """A character in the story with a name and description."""
    name: str
    description: str
    
    @classmethod
    def from_string(cls, text: str) -> 'Character':
        """Parse a character from a string.
        
        Args:
            text: String containing character information
            
        Returns:
            A Character instance
        """
        # Parse character name and description from formatted text
        character_pattern = re.compile(
            rf"{MARKERS['CHARACTER']}(.*?){MARKERS['DESCRIPTION']}(.*?)$",
            re.DOTALL
        )
        match = character_pattern.search(text)
        
        if match:
            name = match.group(1).strip()
            description = match.group(2).strip()
            return cls(name=name, description=description)
        else:
            # Fallback parsing for other formats
            parts = text.split('-', 1)
            if len(parts) == 2:
                name = parts[0].strip()
                description = parts[1].strip()
                return cls(name=name, description=description)
            else:
                # If all else fails, just use the text as the name
                return cls(name=text.strip(), description="")
    
    def to_string(self) -> str:
        """Convert the character to a formatted string.
        
        Returns:
            Formatted character string
        """
        return f"{MARKERS['CHARACTER']}{self.name}{MARKERS['DESCRIPTION']}{self.description}"


@dataclass
class Characters:
    """Collection of characters in the story."""
    character_descriptions: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_string(cls, text: str) -> 'Characters':
        """Parse a collection of characters from a string.
        
        Args:
            text: String containing multiple character entries
            
        Returns:
            A Characters instance
        """
        character_descriptions = {}
        
        # Extract character descriptions using the pattern
        character_pattern = re.compile(
            rf"{MARKERS['CHARACTER']}(.*?){MARKERS['DESCRIPTION']}(.*?)(?={MARKERS['CHARACTER']}|\Z)",
            re.DOTALL
        )
        
        matches = character_pattern.finditer(text)
        for match in matches:
            name = match.group(1).strip()
            description = match.group(2).strip()
            character_descriptions[name] = description
        
        return cls(character_descriptions=character_descriptions)
    
    def to_string(self) -> str:
        """Convert all characters to a formatted string.
        
        Returns:
            Formatted character collection string
        """
        result = ""
        for name, description in self.character_descriptions.items():
            result += f"{MARKERS['CHARACTER']}{name}{MARKERS['DESCRIPTION']}{description}\n\n"
        return result.strip()


@dataclass
class Place:
    """A location in the story with a name and description."""
    name: str
    description: str
    
    @classmethod
    def format_name(cls, name: str) -> str:
        """Format a place name.
        
        Args:
            name: Raw place name
            
        Returns:
            Formatted place name
        """
        return name.strip()
    
    @classmethod
    def from_string(cls, place_name: str, place_text: str) -> 'Place':
        """Create a Place instance from name and description strings.
        
        Args:
            place_name: Name of the place
            place_text: Description of the place
            
        Returns:
            A Place instance
        """
        return cls(name=cls.format_name(place_name), description=place_text.strip())
    
    @classmethod
    def format_prefix(cls, name: str) -> str:
        """Format a place name as a prefix.
        
        Args:
            name: Place name
            
        Returns:
            Formatted place prefix
        """
        return f"{ELEMENTS['PLACE']}{name}"
    
    def to_string(self) -> str:
        """Convert the place to a formatted string.
        
        Returns:
            Formatted place string
        """
        return f"{self.format_prefix(self.name)}\n{self.description}"


@dataclass
class Scene:
    """A scene in the story with a place, plot element, and beat description."""
    place: str
    plot_element: str
    beat: str
    
    def to_string(self) -> str:
        """Convert the scene to a formatted string.
        
        Returns:
            Formatted scene string
        """
        return (
            f"{ELEMENTS['PLACE']}{self.place}\n"
            f"{ELEMENTS['PLOT']}{self.plot_element}\n"
            f"{ELEMENTS['BEAT']}{self.beat}"
        )


@dataclass
class Scenes:
    """Collection of scenes in the story."""
    scenes: List[Scene] = field(default_factory=list)
    
    @classmethod
    def from_string(cls, text: str) -> 'Scenes':
        """Parse a collection of scenes from a string.
        
        Args:
            text: String containing scene descriptions
            
        Returns:
            A Scenes instance
        """
        scenes = []
        
        # Parse places, plot elements, and beats
        places, plot_elements, beats = parse_places_plot_beats(text)
        
        # Create scenes from the parsed elements
        for place, plot_element, beat in zip(places, plot_elements, beats):
            scenes.append(Scene(place=place, plot_element=plot_element, beat=beat))
        
        return cls(scenes=scenes)
    
    def to_string(self) -> str:
        """Convert all scenes to a formatted string.
        
        Returns:
            Formatted scene collection string
        """
        result = ""
        for scene in self.scenes:
            result += scene.to_string() + "\n\n"
        return result.strip()
    
    def num_places(self) -> int:
        """Count the unique places in the scenes.
        
        Returns:
            Number of unique places
        """
        return len(set(scene.place for scene in self.scenes))
    
    def num_scenes(self) -> int:
        """Count the number of scenes.
        
        Returns:
            Number of scenes
        """
        return len(self.scenes)


@dataclass
class Story:
    """Complete story with all components."""
    storyline: str
    title: str
    character_descriptions: Dict[str, str]
    place_descriptions: Dict[str, Place]
    scenes: Scenes
    dialogs: List[str]
    
    def __post_init__(self):
        """Convert title to Title object if it's a string."""
        if isinstance(self.title, str):
            self.title = Title(title=self.title)


def parse_places_plot_beats(text: str) -> tuple[List[str], List[str], List[str]]:
    """Parse places, plot elements, and beats from text.
    
    Args:
        text: Text containing scene descriptions
        
    Returns:
        Tuple of (places, plot_elements, beats)
    """
    # Initialize lists
    places = []
    plot_elements = []
    beats = []
    
    # Get lines from text
    lines = text.split('\n')
    
    current_place = None
    current_plot_element = None
    current_beat = []
    
    for line in lines:
        line = line.strip()
        
        if line.startswith(ELEMENTS['PLACE']):
            # If we were already processing a beat, save it
            if current_place is not None and current_plot_element is not None and current_beat:
                places.append(current_place)
                plot_elements.append(current_plot_element)
                beats.append(' '.join(current_beat))
            
            # Start a new place
            current_place = line[len(ELEMENTS['PLACE']):].strip()
            current_plot_element = None
            current_beat = []
            
        elif line.startswith(ELEMENTS['PLOT']) and current_place is not None:
            current_plot_element = line[len(ELEMENTS['PLOT']):].strip()
            
        elif line.startswith(ELEMENTS['BEAT']) and current_place is not None and current_plot_element is not None:
            # Start collecting the beat
            beat_text = line[len(ELEMENTS['BEAT']):].strip()
            current_beat = [beat_text]
            
        elif current_beat and line and not line.startswith((ELEMENTS['PLACE'], ELEMENTS['PLOT'])):
            # Continue collecting the beat
            current_beat.append(line)
    
    # Don't forget the last beat
    if current_place is not None and current_plot_element is not None and current_beat:
        places.append(current_place)
        plot_elements.append(current_plot_element)
        beats.append(' '.join(current_beat))
    
    return places, plot_elements, beats


def parse_characters_and_descriptions(text: str) -> Dict[str, str]:
    """Parse character names and descriptions from text.
    
    Args:
        text: Text containing character descriptions
        
    Returns:
        Dictionary mapping character names to descriptions
    """
    character_pattern = re.compile(
        rf"{MARKERS['CHARACTER']}(.*?){MARKERS['DESCRIPTION']}(.*?)(?={MARKERS['CHARACTER']}|\Z)",
        re.DOTALL
    )
    
    character_descriptions = {}
    for match in character_pattern.finditer(text):
        name = match.group(1).strip()
        description = match.group(2).strip()
        character_descriptions[name] = description
    
    return character_descriptions


def extract_elements(text: str, begin: str, end: str) -> List[str]:
    """Extract elements from text between begin and end markers.
    
    Args:
        text: Text to extract from
        begin: Beginning marker
        end: Ending marker
        
    Returns:
        List of extracted elements
    """
    elements = []
    start_index = 0
    
    while True:
        begin_index = text.find(begin, start_index)
        if begin_index == -1:
            break
            
        begin_index += len(begin)
        end_index = text.find(end, begin_index)
        
        if end_index == -1:
            # If no end marker is found, use the rest of the text
            element = text[begin_index:].strip()
        else:
            element = text[begin_index:end_index].strip()
            
        elements.append(element)
        
        if end_index == -1:
            break
            
        start_index = end_index + len(end)
    
    return elements


def strip_remove_end(text: str) -> str:
    """Strip whitespace and remove end marker.
    
    Args:
        text: Text to process
        
    Returns:
        Processed text
    """
    text = text.strip()
    if text.endswith(MARKERS['END']):
        text = text[:text.rfind(MARKERS['END'])].strip()
    return text 