"""
Text processing utilities for DramaTron.

This module provides functions for processing and analyzing text.
"""
from typing import Dict, List, Set, Tuple, Optional
import re

from ..config.constants import MARKERS


def detect_repetition_loop(text: str, max_repetitions: int = 4) -> bool:
    """Detect repetition loops in text.
    
    Args:
        text: Text to check for repetitions
        max_repetitions: Maximum number of repetitions to tolerate
        
    Returns:
        True if a repetition loop is detected, False otherwise
    """
    # Split into lines for analysis
    lines = text.splitlines()
    
    # Need at least this many lines to detect a pattern
    min_lines_for_pattern = 6
    
    if len(lines) < min_lines_for_pattern:
        return False
    
    # Try different pattern lengths
    for pattern_length in range(1, min(10, len(lines) // 2)):
        repetition_count = 0
        
        # Start from the end and check backwards
        for i in range(len(lines) - pattern_length, 0, -pattern_length):
            current_chunk = lines[i:i + pattern_length]
            previous_chunk = lines[i - pattern_length:i]
            
            # Check if chunks are equal (simple string comparison)
            if current_chunk == previous_chunk:
                repetition_count += 1
                
                if repetition_count >= max_repetitions:
                    return True
            else:
                break
    
    return False


def strip_markers(text: str) -> str:
    """Remove special markers from text.
    
    Args:
        text: Text to process
        
    Returns:
        Processed text
    """
    text = text.strip()
    
    # Remove end marker
    if text.endswith(MARKERS['END']):
        text = text[:text.rfind(MARKERS['END'])].strip()
        
    return text


def extract_sections(text: str, begin_marker: str, end_marker: str) -> List[str]:
    """Extract sections of text between markers.
    
    Args:
        text: Text to extract from
        begin_marker: Beginning marker
        end_marker: Ending marker
        
    Returns:
        List of extracted sections
    """
    sections = []
    start_index = 0
    
    while True:
        begin_index = text.find(begin_marker, start_index)
        
        if begin_index == -1:
            break
            
        begin_index += len(begin_marker)
        end_index = text.find(end_marker, begin_index)
        
        if end_index == -1:
            # If no end marker is found, use the rest of the text
            section = text[begin_index:].strip()
        else:
            section = text[begin_index:end_index].strip()
            
        sections.append(section)
        
        if end_index == -1:
            break
            
        start_index = end_index + len(end_marker)
    
    return sections


def parse_toxicity_rating(text: str) -> List[float]:
    """Parse toxicity ratings from text.
    
    Args:
        text: Text containing toxicity ratings
        
    Returns:
        List of toxicity ratings as floats
    """
    # Try to extract comma-separated numbers
    number_pattern = re.compile(r'\d+\.\d+|\d+')
    matches = number_pattern.findall(text)
    
    # Convert to floats
    ratings = [float(match) for match in matches]
    
    # Ensure we have at least 5 ratings (pad with zeros if needed)
    while len(ratings) < 5:
        ratings.append(0.0)
        
    # Cap at 5 ratings
    return ratings[:5]


def create_prompt_for_ner(text: str) -> str:
    """Create a prompt for named entity recognition.
    
    Args:
        text: Text to process
        
    Returns:
        Prompt for NER
    """
    return f"""
    Convert the following dialogue text into a list of dictionaries. Each dictionary should represent a character, with the character's name as the key and another dictionary as the value. In this inner dictionary, the keys are sequence numbers based on the overall order of dialogue lines, and the values are the character's dialogue lines. 
    Exclude narrative text, stage directions, and any non-dialogue content.
    For example, given the dialogue:

    Sasha: Hello, my friend!
    Sasha: How are you doing?
    Oleg: Hi. Nothing special, just like everyone else.

    The output should be:

    {{'Sasha': {{1: 'Hello, my friend!', 2: 'How are you doing?'}}}},
    {{'Oleg': {{3: 'Hi. Nothing special, just like everyone else.'}}}}

    Return the results as a plain text.    
    Text for processing: 
    {text}
    """


def clean_dialog_text(text: str) -> str:
    """Clean dialog text, removing stage directions and non-dialog content.
    
    Args:
        text: Dialog text to clean
        
    Returns:
        Cleaned dialog text
    """
    # Remove stage directions in parentheses
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Remove other common stage direction formats
    text = re.sub(r'\[[^]]*\]', '', text)
    text = re.sub(r'\{[^}]*\}', '', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip() 