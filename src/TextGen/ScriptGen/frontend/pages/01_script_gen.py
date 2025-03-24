"""
Script Generation Page

This page provides a user interface for generating new scenes.
"""

import streamlit as st
import logging
from typing import Dict, List, Any, Optional

# Import utilities and components
from ..utils.api import ScriptGenAPI
from ..utils.session import get_state, set_state, add_scene_to_history
from ..utils.formatting import format_screenplay_html
from ..components.llm_selector import llm_selector, llm_config_display

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('script_gen_page')

# Initialize API client
@st.cache_resource
def get_api_client():
    """Get API client singleton."""
    return ScriptGenAPI()


def display_character_selection():
    """Display character selection UI."""
    st.subheader("Characters")
    
    # Get characters from session state
    characters = get_state("bibles", {}).get("character", [])
    
    # If no characters in session state, try to load them
    if not characters:
        api = get_api_client()
        try:
            characters = api.get_all_bible_entries(entry_type="character")
            bibles = get_state("bibles", {})
            bibles["character"] = characters
            set_state("bibles", bibles)
        except Exception as e:
            logger.error(f"Error loading characters: {e}")
            characters = []
    
    # Create dictionary for selected characters
    selected_chars = {}
    
    # Display characters
    if characters:
        # Use columns to display characters in a grid
        cols = st.columns(2)
        for i, char in enumerate(characters):
            char_id = char.get("id", i)
            char_name = char.get("content", {}).get("name", f"Character {char_id}")
            
            with cols[i % 2]:
                selected = st.checkbox(
                    char_name,
                    key=f"char_{char_id}",
                    help=f"Include {char_name} in the scene"
                )
                selected_chars[char_id] = selected
    else:
        st.info("No characters found. Create some characters in the Bible tab first.")
    
    # Store selected character IDs
    selected_char_ids = [char_id for char_id, selected in selected_chars.items() if selected]
    
    return selected_char_ids


def display_plot_selection():
    """Display plot selection UI."""
    st.subheader("Plot")
    
    # Get plots from session state
    plots = get_state("bibles", {}).get("plot", [])
    
    # If no plots in session state, try to load them
    if not plots:
        api = get_api_client()
        try:
            plots = api.get_all_bible_entries(entry_type="plot")
            bibles = get_state("bibles", {})
            bibles["plot"] = plots
            set_state("bibles", bibles)
        except Exception as e:
            logger.error(f"Error loading plots: {e}")
            plots = []
    
    # Add "None" option
    plot_options = [{"id": None, "content": {"title": "None (generate standalone scene)"}}]
    plot_options.extend(plots)
    
    # Display plot selection
    if plots:
        selected_plot_index = st.radio(
            "Select Plot",
            options=range(len(plot_options)),
            format_func=lambda i: plot_options[i]["content"]["title"],
            horizontal=True
        )
        
        selected_plot = plot_options[selected_plot_index]
        selected_plot_id = selected_plot["id"]
    else:
        st.info("No plots found. You can still generate a standalone scene.")
        selected_plot_id = None
    
    return selected_plot_id


def display_scene_form():
    """Display scene generation form."""
    st.header("Generate New Scene")
    
    # Scene ID input
    scene_id = st.text_input(
        "Scene ID",
        value=f"scene_{len(get_state('scenes', {}))+1}",
        help="Unique identifier for this scene"
    )
    
    # Log line input
    log_line = st.text_area(
        "Log Line",
        help="A brief description of the scene to generate",
        height=100
    )
    
    # Character selection
    selected_char_ids = display_character_selection()
    
    # Plot selection
    selected_plot_id = display_plot_selection()
    
    # Generation button
    generate_button = st.button(
        "Generate Scene",
        type="primary",
        disabled=(not log_line or len(selected_char_ids) == 0)
    )
    
    return {
        "scene_id": scene_id,
        "log_line": log_line,
        "character_ids": selected_char_ids,
        "plot_id": selected_plot_id,
        "generate_clicked": generate_button
    }


def display_generated_scene(scene: Dict[str, Any]):
    """
    Display a generated scene.
    
    Args:
        scene: Generated scene data
    """
    st.subheader(scene.get("title") or "Generated Scene")
    
    # Create tabs for different views
    scene_tabs = st.tabs(["Outline", "Character Dialogue", "Integrated Scene"])
    
    # Outline tab
    with scene_tabs[0]:
        outline = scene.get("outline", {})
        
        st.markdown("#### Scene Outline")
        
        if outline:
            st.write(f"**Location:** {outline.get('location', 'Unknown')}")
            st.write(f"**Time:** {outline.get('time', 'Unknown')}")
            st.write(f"**Description:** {outline.get('description', '')}")
            
            if "objectives" in outline and outline["objectives"]:
                st.write("**Objectives:**")
                for obj in outline["objectives"]:
                    st.write(f"- {obj}")
        else:
            st.write("No outline available.")
    
    # Character Dialogue tab
    with scene_tabs[1]:
        dialogue = scene.get("character_dialogue", [])
        
        st.markdown("#### Character Dialogue")
        
        if dialogue:
            for char_dialogue in dialogue:
                char_name = char_dialogue.get("character", "Unknown")
                st.markdown(f"**{char_name}**")
                
                lines = char_dialogue.get("dialogue_lines", [])
                for line in lines:
                    line_char = line.get("character", char_name)
                    action = line.get("action", "")
                    text = line.get("text", "")
                    
                    if action:
                        st.markdown(f"*({action})*  \n{text}")
                    else:
                        st.markdown(text)
                
                st.markdown("---")
        else:
            st.write("No character dialogue available.")
    
    # Integrated Scene tab
    with scene_tabs[2]:
        integrated_scene = scene.get("integrated_scene", "")
        
        st.markdown("#### Integrated Scene")
        
        if integrated_scene:
            st.markdown(format_screenplay_html(integrated_scene), unsafe_allow_html=True)
        else:
            st.write("No integrated scene available.")
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Save Scene"):
            # In a real implementation, this would save to database
            scenes = get_state("scenes", {})
            scenes[scene["scene_id"]] = scene
            set_state("scenes", scenes)
            st.success("Scene saved!")
    
    with col2:
        if integrated_scene:
            st.download_button(
                "Download as Text",
                integrated_scene,
                file_name=f"{scene.get('scene_id', 'scene')}.txt",
                mime="text/plain"
            )


def generate_scene(form_data: Dict[str, Any], llm_config: Dict[str, Any]):
    """
    Generate a scene using the API.
    
    Args:
        form_data: Form data with scene parameters
        llm_config: LLM configuration
        
    Returns:
        Generated scene data
    """
    api = get_api_client()
    
    try:
        with st.spinner("Generating scene..."):
            scene = api.generate_scene(
                scene_id=form_data["scene_id"],
                log_line=form_data["log_line"],
                character_ids=form_data["character_ids"],
                plot_id=form_data["plot_id"],
                llm_config=llm_config
            )
            
            # Add to scene history
            add_scene_to_history(form_data["scene_id"], scene)
            
            return scene
    except Exception as e:
        logger.error(f"Error generating scene: {e}")
        st.error(f"Error generating scene: {str(e)}")
        return None


def script_gen_page():
    """Main function for script generation page."""
    st.title("Generate New Scene")
    
    # LLM selection in sidebar
    with st.sidebar:
        st.markdown("### LLM Configuration")
        llm_config = llm_selector()
    
    # Scene generation form
    form_data = display_scene_form()
    
    # Generate scene if button clicked
    if form_data["generate_clicked"]:
        if not form_data["log_line"]:
            st.error("Please enter a log line first")
        elif not form_data["character_ids"]:
            st.error("Please select at least one character")
        else:
            scene = generate_scene(form_data, llm_config)
            if scene:
                # Store in session state
                set_state("current_scene", scene)
    
    # Display generated scene
    current_scene = get_state("current_scene")
    if current_scene:
        st.markdown("---")
        display_generated_scene(current_scene)


# Run the page
script_gen_page() 