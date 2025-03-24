"""
User interface modules for DramaTron.

This package provides interfaces for interacting with the DramaTron system,
including web-based UIs and command-line tools.
"""
from .session_state import (
    initialize_session_state,
    save_provider,
    save_generator,
    save_story,
    get_provider,
    get_generator,
    get_story,
    clear_session
)

from .export import (
    export_as_text,
    export_as_json,
    export_as_csv,
    export_as_fountain,
    export_as_xml,
    story_to_dict
)

from .multi_model import (
    ModelComparison,
    display_model_comparison
)

# Don't directly import the Streamlit app or CLI modules
# as they are meant to be run as scripts

__all__ = [
    # Session state management
    'initialize_session_state',
    'save_provider',
    'save_generator',
    'save_story',
    'get_provider',
    'get_generator',
    'get_story',
    'clear_session',
    
    # Export utilities
    'export_as_text',
    'export_as_json',
    'export_as_csv',
    'export_as_fountain',
    'export_as_xml',
    'story_to_dict',
    
    # Multi-model comparison
    'ModelComparison',
    'display_model_comparison'
] 