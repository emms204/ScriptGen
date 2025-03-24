"""
Utilities for rendering stories to formatted text.

This module provides functions for rendering story elements into formatted text.
"""
from typing import Dict, List, Optional, Set, Union
import re

from ..models.core import Story, Scene, Place


def render_title(story: Story) -> str:
    """Render the title of a story.
    
    Args:
        story: Story to render title for
        
    Returns:
        Formatted title string
    """
    return f"Title: {story.title}\n"


def render_character_descriptions(story: Story) -> str:
    """Render the character descriptions of a story.
    
    Args:
        story: Story to render character descriptions for
        
    Returns:
        Formatted character descriptions string
    """
    result = "Characters:\n\n"
    
    for name, description in story.character_descriptions.items():
        result += f"{name} - {description}\n\n"
        
    return result


def render_scene(scene: Scene, index: int, story: Story) -> str:
    """Render a scene including its place description.
    
    Args:
        scene: Scene to render
        index: Index of the scene
        story: Story containing the scene
        
    Returns:
        Formatted scene string
    """
    place = story.place_descriptions.get(scene.place)
    place_description = place.description if place else ""
    
    result = f"Scene {index + 1}:\n\n"
    result += f"Place: {scene.place}\n"
    result += f"Plot element: {scene.plot_element}\n"
    result += f"Beat: {scene.beat}\n\n"
    
    result += f"Place Description:\n{place_description}\n\n"
    
    return result


def render_dialog(dialog: str, index: int) -> str:
    """Render a dialog.
    
    Args:
        dialog: Dialog to render
        index: Index of the dialog
        
    Returns:
        Formatted dialog string
    """
    return f"Dialog {index + 1}:\n\n{dialog}\n\n"


def render_story(story: Story) -> str:
    """Render a complete story as a formatted string.
    
    Args:
        story: Story to render
        
    Returns:
        Formatted story string
    """
    result = ""
    
    # Add storyline
    result += f"Storyline: {story.storyline}\n\n"
    
    # Add title
    result += render_title(story)
    result += "\n"
    
    # Add character descriptions
    result += render_character_descriptions(story)
    result += "\n"
    
    # Add scenes and dialogs
    for i, (scene, dialog) in enumerate(zip(story.scenes.scenes, story.dialogs)):
        result += render_scene(scene, i, story)
        result += render_dialog(dialog, i)
        result += "\n"
        
    return result


def character_appears_in_scene(character: str, scene_text: str) -> bool:
    """Check if a character appears in a scene.
    
    Args:
        character: Character name to check
        scene_text: Scene text to check
        
    Returns:
        True if the character appears, False otherwise
    """
    # Simple check if the character name appears in the scene text
    return re.search(rf"\b{re.escape(character)}\b", scene_text) is not None


def place_appears_in_scene(place: str, scene: Scene) -> bool:
    """Check if a place appears in a scene.
    
    Args:
        place: Place name to check
        scene: Scene to check
        
    Returns:
        True if the place appears, False otherwise
    """
    return scene.place == place


def extract_characters_from_dialog(dialog: str) -> Set[str]:
    """Extract character names from a dialog.
    
    Args:
        dialog: Dialog to extract characters from
        
    Returns:
        Set of character names
    """
    # Simple pattern to match character names at the start of lines
    character_pattern = re.compile(r"^([A-Z][a-zA-Z]*(?:\s[A-Z][a-zA-Z]*)?)\s*:", re.MULTILINE)
    
    matches = character_pattern.findall(dialog)
    return set(matches) 