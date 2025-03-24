"""
Web-based UI for DramaTron using Streamlit.

This module provides a user interface for interacting with the DramaTron
text generation system, allowing users to create and edit stories.
"""
import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any

# Add parent directory to path to make imports work
sys.path.append(str(Path(__file__).parent.parent.parent))

from TextGen_refactored import (
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    GroqProvider,
    ProviderFactory,
    StoryGenerator,
    EnhancedStoryGenerator,
    GenerationHistory,
    Story,
    Scene,
    Character,
    render_story
)
from TextGen_refactored.config.constants import DEFAULT_CONFIG
from TextGen_refactored.interface.multi_model import ModelComparison, display_model_comparison

# Set page config
st.set_page_config(
    page_title="DramaTron - Screenplay Generator",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
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
if "model_comparison" not in st.session_state:
    st.session_state.model_comparison = ModelComparison()
if "multi_model_mode" not in st.session_state:
    st.session_state.multi_model_mode = False
if "model_selections" not in st.session_state:
    st.session_state.model_selections = []
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = {}

# App title and description
st.title("🎭 DramaTron")
st.subheader("AI-Powered Screenplay Generator")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Keys
    with st.expander("API Keys", expanded=False):
        openai_key = st.text_input("OpenAI API Key", type="password", 
                                  value=os.environ.get("OPENAI_API_KEY", ""))
        anthropic_key = st.text_input("Anthropic API Key", type="password",
                                     value=os.environ.get("ANTHROPIC_API_KEY", ""))
        gemini_key = st.text_input("Google Gemini API Key", type="password",
                                  value=os.environ.get("GEMINI_API_KEY", ""))
        groq_key = st.text_input("Groq API Key", type="password",
                                value=os.environ.get("GROQ_API_KEY", ""))
        
        if st.button("Save API Keys"):
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
            if anthropic_key:
                os.environ["ANTHROPIC_API_KEY"] = anthropic_key
            if gemini_key:
                os.environ["GEMINI_API_KEY"] = gemini_key
            if groq_key:
                os.environ["GROQ_API_KEY"] = groq_key
            st.success("API keys saved!")
    
    # Multi-model mode toggle
    st.subheader("Model Selection Mode")
    multi_model_mode = st.checkbox("Compare Multiple Models", value=st.session_state.multi_model_mode)
    st.session_state.multi_model_mode = multi_model_mode
    
    if multi_model_mode:
        st.info("In multi-model mode, you can select multiple models to compare their outputs")
        
        # Multi-model selection
        st.subheader("Select Models to Compare")
        
        # OpenAI models
        with st.expander("OpenAI Models", expanded=False):
            openai_models = {
                "GPT-4o": st.checkbox("GPT-4o", key="openai_gpt4o"),
                "GPT-4": st.checkbox("GPT-4", key="openai_gpt4"),
                "GPT-3.5": st.checkbox("GPT-3.5 Turbo", key="openai_gpt35")
            }
        
        # Anthropic models
        with st.expander("Anthropic Models", expanded=False):
            anthropic_models = {
                "Claude-3.5-Sonnet": st.checkbox("Claude 3.5 Sonnet", key="anthropic_c35_sonnet"),
                "Claude-3-Opus": st.checkbox("Claude 3 Opus", key="anthropic_c3_opus"),
                "Claude-3-Sonnet": st.checkbox("Claude 3 Sonnet", key="anthropic_c3_sonnet")
            }
        
        # Groq models (open source)
        with st.expander("Open Source Models (via Groq)", expanded=True):
            groq_models = {
                "Llama-3-70B": st.checkbox("Llama 3 70B", key="groq_llama3_70b"),
                "Llama-3-8B": st.checkbox("Llama 3 8B", key="groq_llama3_8b"),
                "Mixtral-8x7B": st.checkbox("Mixtral 8x7B", key="groq_mixtral"),
                "Mistral-7B": st.checkbox("Mistral 7B", key="groq_mistral"),
                "DeepSeek-LLM": st.checkbox("DeepSeek LLM", key="groq_deepseek"),
                "Gemma-7B": st.checkbox("Gemma 7B", key="groq_gemma"),
                "Qwen2-7B": st.checkbox("Qwen2 7B", key="groq_qwen")
            }
        
        # Google models
        with st.expander("Google Gemini Models", expanded=False):
            gemini_models = {
                "Gemini-1.5-Pro": st.checkbox("Gemini 1.5 Pro", key="gemini_15_pro"),
                "Gemini-1.0-Pro": st.checkbox("Gemini 1.0 Pro", key="gemini_10_pro")
            }
        
        # Collect selected models
        model_selections = []
        
        # OpenAI models
        for name, selected in openai_models.items():
            if selected:
                model_id = f"OpenAI {name}"
                model_info = {
                    "id": model_id,
                    "provider_type": "openai",
                    "model_name": name.lower().replace('-', '') if name != "GPT-3.5" else "gpt-3.5-turbo"
                }
                model_selections.append(model_info)
        
        # Anthropic models
        for name, selected in anthropic_models.items():
            if selected:
                model_id = f"Anthropic {name}"
                model_name = name.lower().replace('-', '-') + "-20240229" 
                if "3.5" in name:
                    model_name = "claude-3-5-sonnet-20240620"
                model_info = {
                    "id": model_id,
                    "provider_type": "anthropic",
                    "model_name": model_name
                }
                model_selections.append(model_info)
        
        # Groq models
        for name, selected in groq_models.items():
            if selected:
                model_id = f"Groq {name}"
                model_name = name.lower().replace('-', '')
                model_info = {
                    "id": model_id,
                    "provider_type": "groq",
                    "model_name": model_name
                }
                model_selections.append(model_info)
        
        # Gemini models
        for name, selected in gemini_models.items():
            if selected:
                model_id = f"Google {name}"
                model_name = name.lower().replace('-', '')
                model_info = {
                    "id": model_id,
                    "provider_type": "gemini",
                    "model_name": model_name
                }
                model_selections.append(model_info)
        
        st.session_state.model_selections = model_selections
        
        # Show selected models
        if model_selections:
            st.success(f"Selected {len(model_selections)} models for comparison")
        else:
            st.warning("Please select at least one model for comparison")
    
    else:
        # Single model selection
        st.subheader("LLM Selection")
        provider_type = st.selectbox(
            "Provider",
            ["OpenAI", "Anthropic", "Google Gemini", "Groq (Open Source Models)"],
            index=0
        )
        
        # Model selection based on provider
        if provider_type == "OpenAI":
            model_name = st.selectbox(
                "Model",
                ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
                index=0
            )
        elif provider_type == "Anthropic":
            model_name = st.selectbox(
                "Model",
                ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-sonnet-20240229"],
                index=0
            )
        elif provider_type == "Google Gemini":
            model_name = st.selectbox(
                "Model",
                ["gemini-1.5-pro", "gemini-1.0-pro"],
                index=0
            )
        else:  # Groq
            model_name = st.selectbox(
                "Model",
                ["llama3-70b", "llama3-8b", "mixtral-8x7b", "mistral-7b", "deepseek-llm", "gemma-7b", "qwen2-7b"],
                index=0
            )
    
    # Generation parameters
    st.subheader("Generation Parameters")
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    max_tokens = st.slider("Max Tokens", min_value=100, max_value=4000, value=1000, step=100)
    
    # Enhanced features
    st.subheader("Advanced Features")
    use_enhanced = st.checkbox("Use Enhanced Generator", value=True)
    filter_toxicity = st.checkbox("Filter Toxicity", value=True)
    generate_storyboards = st.checkbox("Generate Storyboards", value=False)

# Main content area
# Logline input
st.header("Story Outline")
logline = st.text_area(
    "Enter your logline/storyline",
    placeholder="Example: A brilliant but eccentric mathematician discovers an equation that predicts future events, but struggles with the ethical implications when a corporation tries to weaponize the discovery.",
    height=100
)

# Generation scope
col1, col2 = st.columns([1, 1])
with col1:
    generation_scope = st.radio(
        "Generation Scope",
        ["Complete Story", "Single Scene", "Characters Only", "Title Only"]
    )

# Generate button
generate_btn = st.button("Generate")

# Handle multi-model generation
if st.session_state.multi_model_mode and generate_btn and logline:
    if not st.session_state.model_selections:
        st.error("Please select at least one model for comparison")
    else:
        # Clear existing model comparison and initialize with selected models
        model_comparison = ModelComparison()
        
        with st.spinner("Initializing models..."):
            # Add selected models to comparison
            for model_info in st.session_state.model_selections:
                model_id = model_info["id"]
                provider_type = model_info["provider_type"]
                model_name = model_info["model_name"]
                
                # Get API key for the provider
                api_key = None
                if provider_type == "openai":
                    api_key = os.environ.get("OPENAI_API_KEY")
                elif provider_type == "anthropic":
                    api_key = os.environ.get("ANTHROPIC_API_KEY")
                elif provider_type == "gemini":
                    api_key = os.environ.get("GEMINI_API_KEY")
                elif provider_type == "groq":
                    api_key = os.environ.get("GROQ_API_KEY")
                
                # Add model to comparison
                model_comparison.add_model(
                    model_id=model_id,
                    provider_type=provider_type,
                    model_name=model_name,
                    api_key=api_key,
                    config_sampling={"temp": temperature, "prob": 0.9},
                )
        
        # Generate stories with all models
        with st.spinner(f"Generating stories with {len(st.session_state.model_selections)} models..."):
            model_comparison.generate_stories(
                logline=logline,
                enhance=use_enhanced,
                analyze_toxicity=filter_toxicity
            )
            
            # Get results
            comparison_results = model_comparison.get_results()
            st.session_state.comparison_results = comparison_results
            
            # Display model comparison
            display_model_comparison(comparison_results)

# Handle single model generation
elif not st.session_state.multi_model_mode and generate_btn and logline:
    with st.spinner("Initializing language model..."):
        # Map provider type to provider name for factory
        provider_map = {
            "OpenAI": "openai",
            "Anthropic": "anthropic",
            "Google Gemini": "gemini",
            "Groq (Open Source Models)": "groq"
        }
        
        # Create provider using factory
        try:
            factory = ProviderFactory()
            provider = factory.create_provider(
                provider_type=provider_map[provider_type],
                model_name=model_name,
                config_sampling={"prob": 0.9, "temp": temperature},
                default_sample_length=max_tokens
            )
            st.session_state.provider = provider
            
            # Create generator
            if use_enhanced:
                generator = EnhancedStoryGenerator(
                    provider=provider,
                    analyze_toxicity=filter_toxicity
                )
            else:
                generator = StoryGenerator(provider=provider)
            st.session_state.generator = generator
            
            st.success(f"Successfully initialized {provider_type} - {model_name}")
        except Exception as e:
            st.error(f"Error initializing provider: {str(e)}")
            st.info("Check your API keys and try again")

    # Generate content based on scope if we have a generator
    if st.session_state.generator:
        with st.spinner("Generating content..."):
            generator = st.session_state.generator
            
            try:
                if generation_scope == "Complete Story":
                    # Example scene template for generating
                    example_scene = """
                    Place: University Laboratory
                    Plot element: Introduction
                    Beat: Professor works late into the night, finally solving a complex equation
                    
                    Place: Coffee Shop
                    Plot element: Inciting Incident
                    Beat: A corporate representative approaches with an offer to buy the research
                    """
                    
                    story = generator.generate_story(logline)
                    st.session_state.story = story
                    st.session_state.generated = True
                    
                elif generation_scope == "Title Only":
                    title = generator.generate_title()
                    # Create a minimal story with just the title
                    story = Story(
                        storyline=logline,
                        title=title,
                        character_descriptions={},
                        place_descriptions={},
                        scenes=None,
                        dialogs=[]
                    )
                    st.session_state.story = story
                    st.session_state.generated = True
                    
                elif generation_scope == "Characters Only":
                    title = generator.generate_title()
                    characters = generator.generate_characters()
                    # Create a story with title and characters
                    story = Story(
                        storyline=logline,
                        title=title,
                        character_descriptions=characters.character_descriptions,
                        place_descriptions={},
                        scenes=None,
                        dialogs=[]
                    )
                    st.session_state.story = story
                    st.session_state.generated = True
                    
                elif generation_scope == "Single Scene":
                    title = generator.generate_title()
                    characters = generator.generate_characters()
                    
                    # Example scene for generating
                    example_scene = """
                    Place: University Laboratory
                    Plot element: Introduction
                    Beat: Professor works late into the night, finally solving a complex equation
                    """
                    
                    scenes = generator.generate_scenes(example_scene)
                    
                    if scenes and scenes.scenes:
                        # Generate place for first scene
                        place_name = scenes.scenes[0].place
                        place = generator.generate_place(place_name)
                        places = {place_name: place}
                        
                        # Generate dialog for first scene
                        dialog = generator.generate_dialog(0)
                        
                        # Create story with just one scene
                        story = Story(
                            storyline=logline,
                            title=title,
                            character_descriptions=characters.character_descriptions,
                            place_descriptions=places,
                            scenes=scenes,
                            dialogs=[dialog]
                        )
                        st.session_state.story = story
                        st.session_state.generated = True
                    
                st.success("Content generated successfully!")
            except Exception as e:
                st.error(f"Error generating content: {str(e)}")

# Display the generated story if available in single model mode
if not st.session_state.multi_model_mode and st.session_state.generated and st.session_state.story:
    story = st.session_state.story
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Script View", "Scene Editor", "Full Story"])
    
    with tab1:
        # Script view with title and logline
        st.header(story.title.title if hasattr(story.title, 'title') else story.title)
        st.write(f"**Logline:** {story.storyline}")
        
        # Character sidebar
        with st.expander("Characters", expanded=True):
            for name, desc in story.character_descriptions.items():
                st.write(f"**{name}**: {desc}")
        
        # Display scenes if they exist
        if story.scenes and story.scenes.scenes:
            for i, (scene, dialog) in enumerate(zip(story.scenes.scenes, story.dialogs)):
                with st.expander(f"Scene {i+1}: {scene.place} - {scene.plot_element}", expanded=i==0):
                    st.write(f"**Place:** {scene.place}")
                    st.write(f"**Plot Element:** {scene.plot_element}")
                    st.write(f"**Beat:** {scene.beat}")
                    
                    # Display place description if available
                    if scene.place in story.place_descriptions:
                        st.write("---")
                        st.write("**Place Description:**")
                        st.write(story.place_descriptions[scene.place].description)
                    
                    # Display dialog
                    st.write("---")
                    st.write("**Dialog:**")
                    st.text_area(
                        f"Dialog for Scene {i+1}",
                        value=dialog,
                        height=200,
                        key=f"dialog_{i}",
                        disabled=not st.session_state.edit_mode
                    )
    
    with tab2:
        # Scene editor
        st.header("Scene Editor")
        
        if story.scenes and story.scenes.scenes:
            # Scene selection
            scene_index = st.number_input(
                "Scene Number",
                min_value=1,
                max_value=len(story.scenes.scenes),
                value=1
            ) - 1  # Convert to 0-indexed
            
            scene = story.scenes.scenes[scene_index]
            
            # Edit columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Scene details
                st.subheader("Scene Details")
                new_place = st.text_input("Place", value=scene.place)
                new_plot_element = st.text_input("Plot Element", value=scene.plot_element)
                new_beat = st.text_area("Beat", value=scene.beat, height=100)
                
                # Place description
                place_desc = ""
                if scene.place in story.place_descriptions:
                    place_desc = story.place_descriptions[scene.place].description
                
                new_place_desc = st.text_area("Place Description", value=place_desc, height=150)
                
                # Update scene button
                if st.button("Update Scene"):
                    # Update scene in the story
                    generator = st.session_state.generator
                    if generator:
                        try:
                            # Rewrite the scene
                            updated_scenes = generator.rewrite_scene(
                                scene_index,
                                place=new_place,
                                plot_element=new_plot_element,
                                beat=new_beat
                            )
                            
                            # Update place description if changed
                            if new_place != scene.place or new_place_desc != place_desc:
                                generator.rewrite_place(new_place, new_place_desc)
                            
                            # Update story in session state
                            st.session_state.story.scenes = updated_scenes
                            st.success("Scene updated successfully!")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error updating scene: {str(e)}")
            
            with col2:
                # Dialog editor
                st.subheader("Dialog")
                if scene_index < len(story.dialogs):
                    new_dialog = st.text_area(
                        "Edit Dialog",
                        value=story.dialogs[scene_index],
                        height=300
                    )
                    
                    # Update dialog button
                    if st.button("Update Dialog"):
                        generator = st.session_state.generator
                        if generator:
                            try:
                                # Rewrite the dialog
                                updated_dialog = generator.rewrite_dialog(scene_index, new_dialog)
                                
                                # Update story in session state
                                st.session_state.story.dialogs[scene_index] = updated_dialog
                                st.success("Dialog updated successfully!")
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error updating dialog: {str(e)}")
                
                # Regenerate options
                st.subheader("Regenerate")
                
                # Regenerate buttons
                if st.button("Regenerate Scene"):
                    generator = st.session_state.generator
                    if generator:
                        try:
                            # Generate a new scene at the same position
                            example_scene = scene.to_string()
                            updated_scenes = generator.generate_scenes(example_scene)
                            if updated_scenes and updated_scenes.scenes:
                                # Replace just this scene
                                story.scenes.scenes[scene_index] = updated_scenes.scenes[0]
                                st.success("Scene regenerated!")
                                st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error regenerating scene: {str(e)}")
                
                if st.button("Regenerate Dialog"):
                    generator = st.session_state.generator
                    if generator:
                        try:
                            # Generate new dialog for this scene
                            new_dialog = generator.generate_dialog(scene_index)
                            
                            # Update story in session state
                            st.session_state.story.dialogs[scene_index] = new_dialog
                            st.success("Dialog regenerated!")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error regenerating dialog: {str(e)}")
    
    with tab3:
        # Full story view
        st.header("Full Story")
        
        # Render the complete story
        full_story = render_story(story)
        st.text_area("Full Story Text", value=full_story, height=500)
        
        # Export options
        st.download_button(
            label="Download as Text",
            data=full_story,
            file_name="dramatron_story.txt",
            mime="text/plain"
        )

# Footer
st.divider()
st.caption("DramaTron - AI-Powered Screenplay Generator")
st.caption("Built with Streamlit and TextGen_refactored") 