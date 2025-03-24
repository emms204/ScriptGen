"""
Templates for generating different elements of a story.

These templates are used for generating titles, characters, scenes, places, and dialog.
"""
from typing import Dict
from jinja2 import Template


class PromptTemplates:
    """Collection of prompt templates for different story elements."""
    
    TITLE = Template("""
    Using the provided logline: {{ logline }} generate a single title for the story. Finish the generation with **END**
    """.strip())
    
    CHARACTERS = Template("""
    Using the provided logline: {{ logline }}, and title of the story: {{ title }}, create a list of characters which includes their names and their description both for appearance and personal traits.
    Characters and their traits should strictly align with the logline. 
    You can create characters that were not mentioned in the logline, but they have to be logical and improve overall quality of the story.
    Strictly follow the format: 
        **Character:** NAME_OF_THE_CHARACTER **Description:** DESCRIPTION.
    Finish the generation with **END**.
    """.strip())
    
    CHARACTERS_WITH_INITIAL_DATA = Template("""
    Using the provided logline: {{ logline }}, and title of the story: {{ title }}, and initial information about desirable characters: {{ initial_character_data }} create a list of characters which includes their names and their description both for appearance and personal traits.
    Amount of provided characters might be less than you want to generate. You have to use all of the provided descriptions. 
    Characters and their traits should strictly align with the logline. 
    You can create characters that were not mentioned in the logline, but they have to be logical and improve overall quality of the story.
    Strictly follow the format: 
        **Character:** NAME_OF_THE_CHARACTER **Description:** DESCRIPTION.
    Finish the generation with **END**.
    """.strip())
    
    SCENES = Template("""
    Using the provided logline: {{ logline }}, and title of the story: {{ title }}, characters list: {{ character_descriptions }} create a list of scenes for the story. They have to be logical and consistent. 
    Do not invent new characters.
    Strictly follow the format: 
    Place: PLACE_NAME
    Plot element: PLOT_ELEMENT - logical plot element name. 
    Beat: STORY MOMENT
    
    {{ example }}
    """.strip())
    
    SCENES_WITH_INITIAL_DATA = Template("""
    Using the provided logline: {{ logline }}, and title of the story: {{ title }}, characters list: {{ character_descriptions }} and initial scenes information: {{ initial_scenes_data }} create a list of scenes for the story. They have to be logical and consistent. 
    Do not invent new characters.
    Strictly follow the format: 
    Place: PLACE_NAME
    Plot element: PLOT_ELEMENT - logical plot element name. 
    Beat: STORY MOMENT
    
    {{ example }}
    """.strip())
    
    PLACE = Template("""
    Using the provided logline: {{ logline }}, describe the place: {{ place_name }}
    Consider how this location fits into the overall story and what atmosphere it should have. Provide specific details about the setting, including sensory information (sights, sounds, smells).
    Aim for a concise but vivid description that establishes a clear mental image.
    """.strip())
    
    PLACE_WITH_INITIAL_DATA = Template("""
    Using the provided logline: {{ logline }}, and initial place description: {{ initial_place_description }}, enhance and refine the description of the place: {{ place_name }}
    Consider how this location fits into the overall story and what atmosphere it should have. Build upon the initial description, adding specific details about the setting, including sensory information (sights, sounds, smells).
    Aim for a concise but vivid description that establishes a clear mental image.
    """.strip())
    
    DIALOG = Template("""
    Using the provided logline: {{ logline }}, character descriptions: {{ character_descriptions }}, scene: {{ scene }}, and place description: {{ place_description }}, create a realistic and engaging dialogue sequence.
    
    The dialogue should:
    - Stay true to each character's personality and manner of speaking
    - Advance the plot and reflect the scene's purpose
    - Include appropriate stage directions/actions
    - Feel natural and conversational
    - Reveal character relationships and motivations
    
    Format the dialogue with character names followed by their lines, and include brief stage directions in parentheses where needed.
    """.strip())
    
    DIALOG_SEQUENCE = Template("""
    Using the provided logline: {{ logline }}, character descriptions: {{ character_descriptions }}, scene: {{ scene }}, place description: {{ place_description }}, and the previous dialogue: {{ previous_dialog }}, continue the story with a new dialogue sequence.
    
    This dialogue should:
    - Build upon what happened in the previous scene
    - Stay true to each character's personality and manner of speaking
    - Advance the plot and reflect the scene's purpose
    - Include appropriate stage directions/actions
    - Feel natural and conversational
    - Reveal character relationships and motivations
    
    Format the dialogue with character names followed by their lines, and include brief stage directions in parentheses where needed.
    """.strip())
    
    STORYBOARDS = Template("""
    Based on the following script excerpt, create a sequence of storyboard descriptions.
    For each key moment in the script, provide:
    
    1. **Visuals**: Describe what would be seen in a storyboard frame (camera angle, composition, action, etc.)
    2. **Text**: Any important dialogue or text that should accompany this frame
    3. **Characters**: Which characters appear in this frame
    
    Focus on the most important moments that drive the story forward. Create approximately 5-8 storyboard frames.
    
    SCRIPT:
    {{ script_text }}
    """.strip())
    
    TOXICITY = Template("""
    Analyze the following text for potentially harmful content. Rate each category from 0 to 1, where 0 means no harmful content and 1 means severe harmful content.
    
    Categories:
    - Violence
    - Hate speech
    - Sexual content
    - Self-harm
    - Harassment
    
    Provide only numerical ratings for each category, separated by commas, with no additional text.
    
    TEXT:
    {{ text }}
    """.strip())
    
    REGENERATE = Template("""
    The following text potentially contains harmful or inappropriate content. Please rewrite it to remove any problematic elements while preserving the core information and purpose. Make it appropriate for a general audience.
    
    TEXT:
    {{ text }}
    """.strip())


def render_template(template_name: str, **kwargs) -> str:
    """Render a template with the given variables.
    
    Args:
        template_name: Name of the template to render
        **kwargs: Variables to use in the template
        
    Returns:
        Rendered template string
        
    Raises:
        AttributeError: If the template name doesn't exist
    """
    template = getattr(PromptTemplates, template_name.upper())
    return template.render(**kwargs)


def get_all_templates() -> Dict[str, Template]:
    """Get all available templates.
    
    Returns:
        Dictionary mapping template names to Template objects
    """
    return {
        name: value for name, value in PromptTemplates.__dict__.items()
        if isinstance(value, Template) and not name.startswith('_')
    } 