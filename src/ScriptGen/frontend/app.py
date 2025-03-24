"""
ScriptGen - AI-Powered Script Generation Tool

Main Streamlit application entry point.
"""

import os
import streamlit as st
import logging
from typing import Dict, List, Any, Optional

# Import utilities
from .utils.api import ScriptGenAPI
from .utils.session import initialize_session_state, get_state, set_state, update_ui_state
from .utils.formatting import format_screenplay_html

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('scriptgen_app')

# Page configuration
st.set_page_config(
    page_title="ScriptGen",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize API client
@st.cache_resource
def get_api_client():
    """Get API client singleton."""
    return ScriptGenAPI()

# Apply custom CSS
def apply_custom_css():
    """Apply custom CSS styling."""
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        
        h1, h2, h3 {
            color: #1E3A8A;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #1E3A8A;
            color: white;
        }
        
        div[data-testid="stSidebar"] {
            background-color: #f8f9fa;
            border-right: 1px solid #e9ecef;
        }
        
        .sidebar-header {
            margin-top: 0;
            margin-bottom: 0;
            padding: 0.5rem 0;
            border-bottom: 1px solid #e9ecef;
        }
        
        .stButton button {
            width: 100%;
        }
        
        .scene-card {
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
        }
        
        .scene-card:hover {
            border-color: #1E3A8A;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        
        .bible-card {
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
        }
        
        .bible-card:hover {
            border-color: #1E3A8A;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main application."""
    # Initialize session state
    initialize_session_state()
    
    # Apply custom CSS
    apply_custom_css()
    
    # Initialize API client
    api = get_api_client()
    
    # Header
    st.title("ScriptGen")
    st.markdown("AI-Powered Script Generation and Rewriting Tool")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Configuration")
        
        # Project selection (placeholder - would be populated from API)
        project_options = ["My First Script", "Mystery Project", "Sci-Fi Adventure"]
        selected_project = st.selectbox(
            "Project",
            options=project_options,
            index=0
        )
        
        st.markdown("---")
        
        # LLM configuration
        st.markdown("### LLM Settings")
        
        llm_providers = ["OpenAI", "Anthropic", "Local"]
        selected_provider = st.selectbox(
            "Provider",
            options=llm_providers,
            index=0
        )
        
        llm_models = {
            "OpenAI": ["GPT-4", "GPT-3.5"],
            "Anthropic": ["Claude 2", "Claude Instant"],
            "Local": ["Llama 2", "Mistral"]
        }
        
        selected_model = st.selectbox(
            "Model",
            options=llm_models.get(selected_provider, ["None"])
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=1000,
            step=100
        )
        
        # Save LLM config to session state
        llm_config = {
            "provider": selected_provider.lower(),
            "model": selected_model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        set_state("llm_config", llm_config)
        
        st.markdown("---")
        
        # Bible sidebar toggle
        bible_sidebar_expanded = get_state("ui_state", {}).get("bible_sidebar_expanded", True)
        
        if st.checkbox("Show Bibles", value=bible_sidebar_expanded):
            update_ui_state("bible_sidebar_expanded", True)
            st.markdown("### Character Bibles")
            
            # Placeholder for character bibles
            characters = [
                {"id": 1, "name": "Detective Smith", "traits": "Determined, cynical"},
                {"id": 2, "name": "Dr. Sarah Chen", "traits": "Intelligent, secretive"}
            ]
            
            for char in characters:
                with st.expander(char["name"]):
                    st.write(f"**Traits:** {char['traits']}")
                    st.button(f"Edit {char['name']}", key=f"edit_char_{char['id']}")
            
            st.button("Add Character")
            
            st.markdown("### Plot Bibles")
            
            # Placeholder for plot bibles
            plots = [
                {"id": 1, "title": "Murder Mystery", "theme": "Justice vs. Revenge"}
            ]
            
            for plot in plots:
                with st.expander(plot["title"]):
                    st.write(f"**Theme:** {plot['theme']}")
                    st.button(f"Edit {plot['title']}", key=f"edit_plot_{plot['id']}")
            
            st.button("Add Plot")
        else:
            update_ui_state("bible_sidebar_expanded", False)
    
    # Main area tabs
    tabs = ["Generate", "Bibles", "Edit", "Review"]
    tab1, tab2, tab3, tab4 = st.tabs(tabs)
    
    # Tab 1: Generate
    with tab1:
        st.header("Generate New Scene")
        
        # Log line input
        log_line = st.text_area(
            "Log Line",
            help="A brief description of the scene to generate",
            height=100
        )
        
        # Character selection
        st.subheader("Characters")
        char_cols = st.columns(2)
        
        with char_cols[0]:
            char1 = st.checkbox("Detective Smith", value=True)
        
        with char_cols[1]:
            char2 = st.checkbox("Dr. Sarah Chen")
        
        # Plot selection
        st.subheader("Plot")
        plot = st.radio(
            "Select Plot",
            options=["Murder Mystery", "None (generate standalone scene)"]
        )
        
        # Generate button
        if st.button("Generate Scene", type="primary"):
            if not log_line:
                st.error("Please enter a log line first")
            else:
                with st.spinner("Generating scene..."):
                    # In a real app, we would call the API here
                    # For demo purposes, we'll use a placeholder
                    st.session_state["generated_scene"] = {
                        "scene_id": "demo_scene_1",
                        "title": "The Investigation Begins",
                        "outline": {
                            "location": "Police Station - Evidence Room",
                            "time": "Night",
                            "description": "Detective Smith discovers a crucial piece of evidence"
                        },
                        "integrated_scene": """INT. POLICE STATION - EVIDENCE ROOM - NIGHT

Detective Smith hunches over a microscope, examining fibers with intense focus.

DETECTIVE SMITH
(examining evidence bag)
This fiber... it's from our standard-issue uniforms.

He sits back, a troubled expression crossing his face.

DETECTIVE SMITH
(to himself)
An inside job. Just as I suspected."""
                    }
        
        # Display generated scene
        if "generated_scene" in st.session_state:
            scene = st.session_state["generated_scene"]
            
            st.subheader(scene.get("title", "Generated Scene"))
            
            scene_tabs = st.tabs(["Outline", "Integrated Scene"])
            
            with scene_tabs[0]:
                outline = scene.get("outline", {})
                st.write(f"**Location:** {outline.get('location', 'Unknown')}")
                st.write(f"**Time:** {outline.get('time', 'Unknown')}")
                st.write(f"**Description:** {outline.get('description', '')}")
            
            with scene_tabs[1]:
                integrated_scene = scene.get("integrated_scene", "")
                st.markdown(format_screenplay_html(integrated_scene), unsafe_allow_html=True)
            
            st.button("Save Scene")
            st.download_button(
                "Download as Text",
                integrated_scene,
                file_name="scene.txt",
                mime="text/plain"
            )
    
    # Tab 2: Bibles
    with tab2:
        st.header("Bible Management")
        
        bible_types = ["Character", "Plot", "Setting", "Theme"]
        bible_tabs = st.tabs(bible_types)
        
        # Character bibles
        with bible_tabs[0]:
            st.subheader("Character Bibles")
            
            if st.button("Add New Character", key="add_character_main"):
                st.session_state["editing_character"] = True
            
            if st.session_state.get("editing_character", False):
                with st.form("character_form"):
                    name = st.text_input("Name")
                    backstory = st.text_area("Backstory")
                    goals = st.text_area("Goals")
                    fears = st.text_area("Fears")
                    traits = st.text_input("Traits (comma separated)")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        submit = st.form_submit_button("Save")
                    with col2:
                        cancel = st.form_submit_button("Cancel")
                    
                    if submit:
                        # In a real app, we would save to the API here
                        st.success(f"Character '{name}' saved!")
                        st.session_state["editing_character"] = False
                    elif cancel:
                        st.session_state["editing_character"] = False
            
            # Display existing characters
            for char in characters:
                st.markdown(f"""
                <div class="bible-card">
                    <h3>{char['name']}</h3>
                    <p><strong>Traits:</strong> {char['traits']}</p>
                    <div style="display: flex; gap: 10px;">
                        <button style="flex: 1;">Edit</button>
                        <button style="flex: 1;">Delete</button>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Plot bibles
        with bible_tabs[1]:
            st.subheader("Plot Bibles")
            
            if st.button("Add New Plot", key="add_plot_main"):
                st.session_state["editing_plot"] = True
            
            if st.session_state.get("editing_plot", False):
                with st.form("plot_form"):
                    title = st.text_input("Title")
                    theme = st.text_input("Theme")
                    main_conflict = st.text_area("Main Conflict")
                    act_structure = st.selectbox(
                        "Act Structure",
                        options=["3-act", "5-act", "Hero's Journey", "Other"]
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        submit = st.form_submit_button("Save")
                    with col2:
                        cancel = st.form_submit_button("Cancel")
                    
                    if submit:
                        # In a real app, we would save to the API here
                        st.success(f"Plot '{title}' saved!")
                        st.session_state["editing_plot"] = False
                    elif cancel:
                        st.session_state["editing_plot"] = False
            
            # Display existing plots
            for plot in plots:
                st.markdown(f"""
                <div class="bible-card">
                    <h3>{plot['title']}</h3>
                    <p><strong>Theme:</strong> {plot['theme']}</p>
                    <div style="display: flex; gap: 10px;">
                        <button style="flex: 1;">Edit</button>
                        <button style="flex: 1;">Delete</button>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Tab 3: Edit
    with tab3:
        st.header("Edit Scenes")
        
        # Placeholder for scenes
        scenes = [
            {"id": "scene_1", "title": "The Investigation Begins", "description": "Detective Smith discovers evidence"},
            {"id": "scene_2", "title": "The Confrontation", "description": "Smith confronts a suspect"}
        ]
        
        selected_scene = st.selectbox(
            "Select Scene",
            options=[scene["title"] for scene in scenes]
        )
        
        # Display selected scene for editing
        scene_idx = next((i for i, scene in enumerate(scenes) if scene["title"] == selected_scene), 0)
        scene = scenes[scene_idx]
        
        st.subheader(scene["title"])
        st.write(scene["description"])
        
        scene_text = """INT. POLICE STATION - EVIDENCE ROOM - NIGHT

Detective Smith hunches over a microscope, examining fibers with intense focus.

DETECTIVE SMITH
(examining evidence bag)
This fiber... it's from our standard-issue uniforms.

He sits back, a troubled expression crossing his face.

DETECTIVE SMITH
(to himself)
An inside job. Just as I suspected."""

        edited_scene_text = st.text_area(
            "Edit Scene",
            value=scene_text,
            height=300
        )
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Save Changes"):
                st.success("Scene saved!")
        
        with col2:
            if st.button("Critique Scene"):
                with st.spinner("Analyzing scene..."):
                    # In a real app, we would call the API here
                    st.info("Scene critique would be displayed here")
        
        with col3:
            if st.button("Rewrite"):
                st.text_area("Rewrite Instructions", value="", placeholder="e.g., 'Make the dialogue more tense'")
                if st.button("Apply Rewrite"):
                    with st.spinner("Rewriting scene..."):
                        # In a real app, we would call the API here
                        st.info("Rewritten scene would be displayed here")
    
    # Tab 4: Review
    with tab4:
        st.header("Review & Feedback")
        
        st.subheader("Dramaturge Critique")
        
        if st.button("Request Critique"):
            with st.spinner("Generating critique..."):
                # In a real app, we would call the API here
                critique = """
                ## Overall Assessment
                The scene effectively establishes the discovery of key evidence but could benefit from more tension and character development.
                
                ## Pacing
                - The scene moves too quickly to the revelation
                - Consider adding a moment of frustration before the discovery
                
                ## Tension
                - The characters' reactions could be more pronounced
                - Add physical tension cues
                
                ## Suggested Rewrites
                1. Add more buildup before the key evidence is found
                2. Include more physical descriptions of Smith's reaction
                """
                st.markdown(critique)
        
        st.subheader("Bible Change Preview")
        
        st.write("Preview how changes to bibles would affect scenes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            bible_type = st.selectbox(
                "Bible Type",
                options=["Character", "Plot", "Setting", "Theme"]
            )
        
        with col2:
            bible_name = st.selectbox(
                "Entry",
                options=["Detective Smith", "Dr. Sarah Chen"] if bible_type == "Character" else ["Murder Mystery"]
            )
        
        change_field = st.selectbox(
            "Field to Change",
            options=["Name", "Traits", "Goals"] if bible_type == "Character" else ["Title", "Theme"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            old_value = st.text_input("Old Value", value="Detective Smith" if change_field == "Name" else "")
        
        with col2:
            new_value = st.text_input("New Value", value="")
        
        if st.button("Preview Changes"):
            with st.spinner("Generating preview..."):
                # In a real app, we would call the API here
                st.info("2 scenes would be affected by this change")
                
                with st.expander("Scene 1: The Investigation Begins"):
                    st.markdown("""
                    ```diff
                    - Detective Smith hunches over a microscope
                    + Detective Johnson hunches over a microscope
                    
                    - DETECTIVE SMITH
                    + DETECTIVE JOHNSON
                    (examining evidence bag)
                    ```
                    """)
                
                with st.expander("Scene 2: The Confrontation"):
                    st.markdown("""
                    ```diff
                    - Smith confronts the suspect
                    + Johnson confronts the suspect
                    
                    - DETECTIVE SMITH
                    + DETECTIVE JOHNSON
                    I know what you did.
                    ```
                    """)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Apply All Changes"):
                        st.success("Changes applied to all scenes!")
                
                with col2:
                    if st.button("Select Scenes to Update"):
                        st.info("Scene selection UI would appear here")

if __name__ == "__main__":
    main() 