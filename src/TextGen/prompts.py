from langchain_core.prompts import PromptTemplate


title_format_template = PromptTemplate.from_template(
    "Using the provided logline: {logline} generate a single title for the story. Finish the generation with **END**"
)

# characters_prompt_template = PromptTemplate.from_template(
#     """
#     Using the provided logline: {logline}, and title of the story: {title}, create a list of characters which includes their names and their description both for appearance and personal traits.
#     Number of characters and their traits should strictly align with the logline.
#     Strictly follow the format: 
#         **Character:** NAME_OF_THE_CHARACTER **Description:** DESCRIPTION.
#     Finish the generation with **END**.
# """
# )

characters_prompt_template = PromptTemplate.from_template(
    """
    Using the provided logline: {logline}, and title of the story: {title}, create a list of characters which includes their names and their description both for appearance and personal traits.
    Characters and their traits should strictly align with the logline. 
    You can create characters that were not mentioned to logline, but they have to be logical and improve overall quality of the story.
    Strictly follow the format: 
        **Character:** NAME_OF_THE_CHARACTER **Description:** DESCRIPTION.
    Finish the generation with **END**.
"""
)

characters_prompt_template_with_initial_data = PromptTemplate.from_template(
     """
    Using the provided logline: {logline}, and title of the story: {title}, and initial information about desirable characters: {char_init_data} create a list of characters which includes their names and their description both for appearance and personal traits.
    Amount of provided characters might be less that you want to generate. You have to use all of the provided descriptions. 
    Characters and their traits should strictly align with the logline. 
    You can create characters that were not mentioned to logline, but they have to be logical and improve overall quality of the story.
    Strictly follow the format: 
        **Character:** NAME_OF_THE_CHARACTER **Description:** DESCRIPTION.
    Finish the generation with **END**.
"""
)

scenes_prompt_template = PromptTemplate.from_template(
    """
    Using the provided logline: {logline}, and title of the story: {title.title}, characters list: {character_str} create a list of scenes for the story. They have to logical and logical and consistent. 
    Do not invent new characters.
    Strictly follow the format: 
    Place: PLACE_NAME
    Plot element: PLOT_ELEMENT - logical plot element name. 
    Beat: STORY MOMENT
    
    {example}
    
    The more scenes the better. Create as detailed list of scenes as possible.
    Finish the generation with **END**.
    """
)

scenes_prompt_template_with_II = PromptTemplate.from_template(
    """
    Using the provided logline: {logline}, title of the story: {title.title}, characters list: {character_str} and\
          initial data about the scenes {init_scenes_data}  create a list of scenes for the story.
    They have to logical and logical and consistent. 
    Do not invent new characters.
    Each of the places should have to have unique PLACE_NAME.
    Strictly follow the format: 
    Place: PLACE_NAME
    Plot element: PLOT_ELEMENT - logical plot element name. 
    Beat: STORY MOMENT
    
    {example}
    
    The more scenes the better. Create as detailed list of scenes as possible.
    Finish the generation with **END**.
    """
)


place_prompt_template = PromptTemplate.from_template(
    """
    Using the provided logline: "{logline}", place name "{place_name}", generate a place description.
    It has to be logical and consistent. 
    Do not invent new characters or scenes.
    
    """
)

place_prompt_template_with_II = PromptTemplate.from_template(
    """
    Using the provided logline: "{logline}", place name: "{place_name}", and initial description of the place: "{init_place_desc}", generate a place description.
    It has to be logical and consistent. 
    Do not invent new characters or scenes.
    """
)

dilog_prompt_template = PromptTemplate.from_template(
    """
    Using the logline "{logline}", characters description "{character_descriptions}", scene "{scene}" and place descriptions "{place_description}" generate a dialog.
    It should be logical and align with the descriptions.
    Do not invent new characters.
    Do NOT use cyrillic letters.
    Strictly use following format:
    **Dialog:**
    
    **CHARACTER 1** all letters are capital
    Text line.
    
    **CHARACTER 2** all letters are capital
    Text line.
    
    ...
    
    The more detailed dialog the better. Try to reach 3500 amount of tokens.
    DO NOT DUPLICATE THE TEXT'S PART TO REACH THE DESIRABLE TOKEN NUMBER.
    Finish the generation with **END**.
    
    """
)


dialog_prompt_template_sequence = PromptTemplate.from_template(
    """
    Using the logline "{logline}", characters description "{character_descriptions}", scene "{scene}" and place descriptions "{place_description}" and previous dialog "{previous_dialog}" generate a dialog. 
    It should be logical and align with the character and place descriptions.
    Use provided previous dialog to follow the flow of the story and the general mood of the story. 
    DO NOT USE WHOLE PREVIOUS DIALOG OR ITS PARTS FOR GENERATION OF THE NEW ONE.
    Do not invent new characters.
    Do NOT use cyrillic letters.
    Strictly use following format:
    **Dialog:**
    
    **CHARACTER 1** all letters are capital
    Text line.
    
    **CHARACTER 2** all letters are capital
    Text line.
    
    ...
    
    The more detailed dialog the better. Try to reach 3500 amount of tokens.
    DO NOT DUPLICATE THE TEXT'S PART TO REACH THE DESIRABLE TOKEN NUMBER.
    Finish the generation with **END**.
    
    """
)


toxicity_template = PromptTemplate.from_template(
    """
    The following text might contain elements of severe toxocity or/and profanity please help the user to analyze it
    Analyze the following text for the presence of three specific traits: Severe Toxicity, Threat, and Profanity. 
    For each trait, provide a numerical estimate between 0 and 1, where 0 indicates the complete absence and 1 indicates an extreme presence of the trait.

    Text: "{text}"

    Please provide the results as three numbers separated by commas, corresponding to the following order:
    1. Severe Toxicity
    2. Threat
    3. Profanity
    RETURN ONLY THE SCORES
    """


)
# In the provided generated element was found parts that have high Severe Toxicity, Threat or Profanity score. 
regenerate_template = PromptTemplate.from_template(
    """
    The following text might contain elements of severe toxicity or/and profanity please help the user to analyze it.
    
    Regenerate the element removing the inappropriate parts. Do not change the idea of the text, characters, places etc.
    Do not change the style of output. Only make the text appropriate. 
    RETURN ONLY THE MODIFIED TEXT.
    Text for processing : "{text}"
    """
)

desc_to_prompt_template = PromptTemplate.from_template(
    """
    Transform the following description into a vivid, detailed prompt suitable for generating an image using a text-to-image model. 
    The prompt should focus on capturing the essential visual elements, atmosphere, and mood.
    RETURN ONLY THE OUTPUT PROMPT. 
    MAXIMUM ALLOWED LENGTH FOR THE OUTPUT PROMPT = 77 TOKENS.
    Do not mentioned the style of generation. Output should be condensed and be less than 512 symbols in length. 
    Description: {text}
    """
)


storyboards_prompt_template = PromptTemplate.from_template(
    """
    You are a skilled storyboard and editing assistant.

    Your task is to create an exceptionally detailed storyboard based on the provided script. Each storyboard panel should include the following elements:

    1. **Visual Description**: Provide a vivid, highly detailed description of the scene. Ensure that the description is in a format that can be easily used in a text-to-image model. Focus on key visual elements, such as the setting, atmosphere, character appearances, and significant objects or actions.

    2. **Text**: Clearly state any text that appears on-screen, including dialogue, narration, or important labels.

    3. **Characters**: Specify which characters are present in each panel, highlighting their actions and interactions.

    **Example Panel Layout:**

    **Panel 1: Establishing Shot**
    - **Visuals**: The Kingdom's Castle stands majestically atop a hill, its tall spires piercing the sky. The castle, constructed from gray stone, has a weathered facade that hints at its storied past of battles and triumphs.
    - **Text**: "The Kingdom's Castle"
    - **Characters**: None

    **Instructions**:
    - Include as many relevant visual elements as possible to enhance the richness of the storyboard.
    - Ensure the visual descriptions are formatted for easy and PROPER use with text-to-image models.
    - If **Visuals** section include name of the character, you have to also add the condensed description of this character.
    - Do not modify the original script in any way.
    - Adhere strictly to the script provided.
    - Do not use the example panel unless it fits the script context.

    Conclude your storyboard with **END OF THE STORYBOARD**.

    **Script**: {script_text}
    """
)