"""
Session state management for Streamlit application.

This module provides utilities for managing session state in the Streamlit application.
"""

import streamlit as st
from typing import Dict, List, Any, Optional, Callable, TypeVar, Generic

T = TypeVar('T')


def get_state_key(key: str) -> str:
    """
    Get a session state key with proper prefix.
    
    Args:
        key: Base key name
        
    Returns:
        Prefixed key
    """
    return f"scriptgen_{key}"


def initialize_session_state() -> None:
    """Initialize the session state with default values."""
    
    # Core state
    if get_state_key("current_project") not in st.session_state:
        st.session_state[get_state_key("current_project")] = None
        
    if get_state_key("current_scene") not in st.session_state:
        st.session_state[get_state_key("current_scene")] = None
    
    # LLM configuration
    if get_state_key("llm_config") not in st.session_state:
        st.session_state[get_state_key("llm_config")] = {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000
        }
    
    # Scene cache
    if get_state_key("scenes") not in st.session_state:
        st.session_state[get_state_key("scenes")] = {}
    
    # Bible entries cache
    if get_state_key("bibles") not in st.session_state:
        st.session_state[get_state_key("bibles")] = {
            "character": [],
            "plot": [],
            "setting": [],
            "theme": []
        }
    
    # Change tracking
    if get_state_key("bible_changes") not in st.session_state:
        st.session_state[get_state_key("bible_changes")] = []
    
    # Scene history for undo/redo
    if get_state_key("scene_history") not in st.session_state:
        st.session_state[get_state_key("scene_history")] = {}
    
    # UI state
    if get_state_key("ui_state") not in st.session_state:
        st.session_state[get_state_key("ui_state")] = {
            "bible_sidebar_expanded": True,
            "rewrite_panel_expanded": False,
            "active_tab": "generate"
        }


def get_state(key: str, default: Optional[T] = None) -> T:
    """
    Get a value from session state.
    
    Args:
        key: State key
        default: Default value if key doesn't exist
        
    Returns:
        Value from session state or default
    """
    state_key = get_state_key(key)
    if state_key not in st.session_state:
        return default
    return st.session_state[state_key]


def set_state(key: str, value: Any) -> None:
    """
    Set a value in session state.
    
    Args:
        key: State key
        value: Value to set
    """
    st.session_state[get_state_key(key)] = value


def clear_state(key: str) -> None:
    """
    Clear a value from session state.
    
    Args:
        key: State key to clear
    """
    state_key = get_state_key(key)
    if state_key in st.session_state:
        del st.session_state[state_key]


def update_ui_state(key: str, value: Any) -> None:
    """
    Update a specific UI state value.
    
    Args:
        key: UI state key
        value: New value
    """
    ui_state = get_state("ui_state", {})
    ui_state[key] = value
    set_state("ui_state", ui_state)


def get_ui_state(key: str, default: Optional[T] = None) -> T:
    """
    Get a specific UI state value.
    
    Args:
        key: UI state key
        default: Default value if key doesn't exist
        
    Returns:
        UI state value
    """
    ui_state = get_state("ui_state", {})
    return ui_state.get(key, default)


def add_scene_to_history(scene_id: str, scene_data: Dict[str, Any]) -> None:
    """
    Add a scene version to history.
    
    Args:
        scene_id: Scene identifier
        scene_data: Scene data to add to history
    """
    history = get_state("scene_history", {})
    
    # Initialize history for this scene if it doesn't exist
    if scene_id not in history:
        history[scene_id] = []
    
    # Add the new version (with timestamp)
    import datetime
    version = {
        "timestamp": datetime.datetime.now().isoformat(),
        "data": scene_data
    }
    
    # Limit history to 10 versions
    scene_history = history[scene_id]
    scene_history.append(version)
    if len(scene_history) > 10:
        scene_history.pop(0)
    
    # Update history in session state
    history[scene_id] = scene_history
    set_state("scene_history", history)


def get_scene_history(scene_id: str) -> List[Dict[str, Any]]:
    """
    Get history for a specific scene.
    
    Args:
        scene_id: Scene identifier
        
    Returns:
        List of scene versions
    """
    history = get_state("scene_history", {})
    return history.get(scene_id, [])


def add_bible_change(
    bible_id: int,
    bible_type: str,
    field_path: str,
    old_value: Any,
    new_value: Any
) -> None:
    """
    Track a bible change for potential propagation.
    
    Args:
        bible_id: Bible entry ID
        bible_type: Type of bible (character, plot, etc.)
        field_path: Path to the changed field
        old_value: Previous value
        new_value: New value
    """
    changes = get_state("bible_changes", [])
    
    # Add the change
    changes.append({
        "bible_id": bible_id,
        "bible_type": bible_type,
        "field_path": field_path,
        "old_value": old_value,
        "new_value": new_value,
        "timestamp": __import__("datetime").datetime.now().isoformat()
    })
    
    # Update changes in session state
    set_state("bible_changes", changes)


def get_bible_changes() -> List[Dict[str, Any]]:
    """
    Get tracked bible changes.
    
    Returns:
        List of bible changes
    """
    return get_state("bible_changes", [])


def clear_bible_changes() -> None:
    """Clear tracked bible changes."""
    set_state("bible_changes", [])


def with_state_loading(key: str, loading_state: str = "loading"):
    """
    Decorator for functions that should update a loading state.
    
    Args:
        key: State key for tracking loading status
        loading_state: Loading state value
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Set loading state
            set_state(key, loading_state)
            try:
                # Call the function
                result = func(*args, **kwargs)
                # Clear loading state on success
                set_state(key, "done")
                return result
            except Exception as e:
                # Set error state on exception
                set_state(key, f"error: {str(e)}")
                raise
        return wrapper
    return decorator 