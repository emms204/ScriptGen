"""
Utility functions for DramaTron.

This package provides utility functions for text processing, rendering, and more.
"""

from .rendering import (
    render_title,
    render_character_descriptions,
    render_scene,
    render_dialog,
    render_story,
    character_appears_in_scene,
    place_appears_in_scene,
    extract_characters_from_dialog
)

from .text_processing import (
    detect_repetition_loop,
    strip_markers,
    extract_sections,
    parse_toxicity_rating,
    create_prompt_for_ner,
    clean_dialog_text
)

__all__ = [
    # Rendering
    'render_title',
    'render_character_descriptions',
    'render_scene',
    'render_dialog',
    'render_story',
    'character_appears_in_scene',
    'place_appears_in_scene',
    'extract_characters_from_dialog',
    
    # Text processing
    'detect_repetition_loop',
    'strip_markers',
    'extract_sections',
    'parse_toxicity_rating',
    'create_prompt_for_ner',
    'clean_dialog_text'
] 