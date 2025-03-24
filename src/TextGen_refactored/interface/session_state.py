"""
Session state management for the Streamlit interface.

This module provides helper functions to manage and initialize the Streamlit session state.
"""
from typing import Dict, Any, Optional
import streamlit as st

from TextGen_refactored.generators import GenerationHistory
from TextGen_refactored.providers import LanguageModelProvider
from TextGen_refactored.generators import StoryGenerator


def initialize_session_state() -> None:
    """Initialize session state with default values if not already present."""
    if "story" not in st.session_state:
        st.session_state.story = None
        
    if "provider" not in st.session_state:
        st.session_state.provider = None
        
    if "generator" not in st.session_state:
        st.session_state.generator = None
        
    if "generated" not in st.session_state:
        st.session_state.generated = False
        
    if "edit_mode" not in st.session_state:
        st.session_state.edit_mode = False
        
    if "scene_index" not in st.session_state:
        st.session_state.scene_index = 0
        
    if "history" not in st.session_state:
        st.session_state.history = GenerationHistory()


def save_provider(provider: LanguageModelProvider) -> None:
    """Save a language model provider to the session state.
    
    Args:
        provider: The language model provider to save
    """
    st.session_state.provider = provider


def save_generator(generator: StoryGenerator) -> None:
    """Save a story generator to the session state.
    
    Args:
        generator: The story generator to save
    """
    st.session_state.generator = generator


def save_story(story: Any) -> None:
    """Save a story to the session state.
    
    Args:
        story: The story to save
    """
    st.session_state.story = story
    st.session_state.generated = True


def get_provider() -> Optional[LanguageModelProvider]:
    """Get the current language model provider from the session state.
    
    Returns:
        The language model provider or None if not set
    """
    return st.session_state.provider if "provider" in st.session_state else None


def get_generator() -> Optional[StoryGenerator]:
    """Get the current story generator from the session state.
    
    Returns:
        The story generator or None if not set
    """
    return st.session_state.generator if "generator" in st.session_state else None


def get_story() -> Optional[Any]:
    """Get the current story from the session state.
    
    Returns:
        The story or None if not set
    """
    return st.session_state.story if "story" in st.session_state else None


def clear_session() -> None:
    """Clear the current session state."""
    if "story" in st.session_state:
        st.session_state.story = None
        
    if "provider" in st.session_state:
        st.session_state.provider = None
        
    if "generator" in st.session_state:
        st.session_state.generator = None
        
    if "generated" in st.session_state:
        st.session_state.generated = False 