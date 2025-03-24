"""
DramaTron: A hierarchical language model-based script generation system.

DramaTron is a tool for generating coherent scripts and screenplays by breaking down
the writing process into structured, hierarchical steps. It uses large language models
to generate text at each level of the hierarchy, with each level building upon the previous one.

The hierarchy consists of:
1. Storyline/Logline - A brief summary of the central dramatic conflict
2. Title - Generated from the logline
3. Characters - Generated from the logline and title
4. Scenes - Generated from the logline, title, and characters
5. Places - Generated from the scenes
6. Dialogs - Generated from the scenes, characters, and places

This modular approach helps maintain coherence across the entire script.
"""

from .models import (
    Title,
    Character,
    Characters,
    Scene,
    Scenes,
    Place,
    Story
)

from .generators import (
    StoryGenerator,
    EnhancedStoryGenerator,
    GenerationHistory
)

from .providers import (
    LanguageModelProvider,
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    GroqProvider,
    ProviderFactory
)

from .utils import (
    render_story,
    render_title,
    render_character_descriptions,
    render_scene,
    render_dialog
)

from .prompts import (
    PromptTemplates,
    render_template
)

# Import interface modules - these are used when running the Streamlit app
# but not imported directly to avoid Streamlit dependencies for non-UI usage
# from .interface import (
#     initialize_session_state,
#     save_provider,
#     save_generator,
#     save_story,
#     get_provider,
#     get_generator,
#     get_story,
#     clear_session,
#     export_as_text,
#     export_as_json,
#     export_as_csv,
#     export_as_fountain,
#     export_as_xml,
#     story_to_dict
# )

__version__ = '1.0.0'
__author__ = 'DramaTron Team'

__all__ = [
    # Models
    'Title',
    'Character',
    'Characters',
    'Scene',
    'Scenes',
    'Place',
    'Story',
    
    # Generators
    'StoryGenerator',
    'EnhancedStoryGenerator',
    'GenerationHistory',
    
    # Providers
    'LanguageModelProvider',
    'OpenAIProvider',
    'AnthropicProvider',
    'GeminiProvider',
    'GroqProvider',
    'ProviderFactory',
    
    # Utils
    'render_story',
    'render_title',
    'render_character_descriptions',
    'render_scene',
    'render_dialog',
    
    # Prompts
    'PromptTemplates',
    'render_template'
] 