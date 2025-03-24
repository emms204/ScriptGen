"""
Multi-model comparison functionality for Dramatron.

This module provides utilities for running multiple language models in parallel
and comparing their outputs.
"""
import threading
import time
from typing import Dict, List, Any, Optional, Tuple

import streamlit as st

from TextGen_refactored.providers import LanguageModelProvider, ProviderFactory
from TextGen_refactored.generators import StoryGenerator, EnhancedStoryGenerator
from TextGen_refactored.models import Story


class ModelComparison:
    """Class for comparing multiple models."""
    
    def __init__(self):
        """Initialize model comparison."""
        self.providers = {}
        self.generators = {}
        self.results = {}
        self.errors = {}
    
    def add_model(
        self, 
        model_id: str, 
        provider_type: str, 
        model_name: str, 
        api_key: Optional[str] = None,
        config_sampling: Optional[Dict[str, Any]] = None,
        enhance: bool = False,
        analyze_toxicity: bool = False
    ) -> None:
        """Add a model to the comparison.
        
        Args:
            model_id: A unique identifier for this model
            provider_type: Type of provider (e.g., "openai", "anthropic", "groq")
            model_name: Name of the model
            api_key: API key for the provider
            config_sampling: Configuration for sampling
            enhance: Whether to use the enhanced generator
            analyze_toxicity: Whether to analyze for toxicity
        """
        try:
            # Create provider factory
            factory = ProviderFactory()
            
            # Create provider
            provider = factory.create_provider(
                provider_type=provider_type,
                model_name=model_name,
                api_key=api_key,
                config_sampling=config_sampling or {"temp": 0.7, "prob": 0.9}
            )
            
            # Store provider
            self.providers[model_id] = provider
            
            # Results and errors will be stored later
            self.results[model_id] = None
            self.errors[model_id] = None
        
        except Exception as e:
            # Store error
            self.errors[model_id] = str(e)
    
    def _generate_story_with_model(
        self, 
        model_id: str, 
        logline: str, 
        enhance: bool, 
        analyze_toxicity: bool
    ) -> None:
        """Generate a story with a model.
        
        Args:
            model_id: Model identifier
            logline: Logline for the story
            enhance: Whether to use the enhanced generator
            analyze_toxicity: Whether to analyze for toxicity
        """
        try:
            provider = self.providers.get(model_id)
            if not provider:
                return
            
            # Create generator
            if enhance:
                generator = EnhancedStoryGenerator(
                    provider=provider,
                    analyze_toxicity=analyze_toxicity
                )
            else:
                generator = StoryGenerator(provider=provider)
            
            # Store generator
            self.generators[model_id] = generator
            
            # Generate story
            story = generator.generate_story(logline)
            
            # Store result
            self.results[model_id] = story
        
        except Exception as e:
            # Store error
            self.errors[model_id] = str(e)
    
    def generate_stories(
        self, 
        logline: str, 
        enhance: bool = False, 
        analyze_toxicity: bool = False
    ) -> None:
        """Generate stories using all configured models.
        
        Args:
            logline: Logline for the story
            enhance: Whether to use the enhanced generator
            analyze_toxicity: Whether to analyze for toxicity
        """
        # Clear previous results and errors
        for model_id in self.providers:
            self.results[model_id] = None
            self.errors[model_id] = None
        
        # Create threads for each model
        threads = []
        for model_id in self.providers:
            thread = threading.Thread(
                target=self._generate_story_with_model,
                args=(model_id, logline, enhance, analyze_toxicity)
            )
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
    
    def get_results(self) -> Dict[str, Tuple[Optional[Story], Optional[str]]]:
        """Get results from all models.
        
        Returns:
            Dictionary mapping model IDs to (story, error) tuples
        """
        return {
            model_id: (self.results.get(model_id), self.errors.get(model_id))
            for model_id in self.providers
        }


def display_model_comparison(comparison_results: Dict[str, Tuple[Optional[Story], Optional[str]]]) -> None:
    """Display a comparison of model outputs in Streamlit.
    
    Args:
        comparison_results: Results from ModelComparison.get_results()
    """
    # Create tabs for each model
    if not comparison_results:
        st.warning("No models to compare")
        return
    
    # Create tabs for each model
    tabs = st.tabs(list(comparison_results.keys()))
    
    # Display results in each tab
    for i, (model_id, (story, error)) in enumerate(comparison_results.items()):
        with tabs[i]:
            if error:
                st.error(f"Error with {model_id}: {error}")
            elif story:
                # Display story
                st.header(getattr(story.title, "title", story.title))
                st.subheader(f"Logline: {story.storyline}")
                
                # Display characters
                if story.character_descriptions:
                    with st.expander("Characters", expanded=True):
                        for name, desc in story.character_descriptions.items():
                            st.write(f"**{name}**: {desc}")
                
                # Display scenes
                if story.scenes and story.scenes.scenes:
                    for j, (scene, dialog) in enumerate(zip(story.scenes.scenes, story.dialogs)):
                        with st.expander(f"Scene {j+1}: {scene.place}", expanded=(j==0)):
                            st.write(f"**Place:** {scene.place}")
                            st.write(f"**Plot Element:** {scene.plot_element}")
                            st.write(f"**Beat:** {scene.beat}")
                            
                            # Display place description
                            if scene.place in story.place_descriptions:
                                place = story.place_descriptions[scene.place]
                                st.write(f"**Place Description:** {place.description}")
                            
                            # Display dialog
                            st.write("**Dialog:**")
                            st.text_area(
                                label="",
                                value=dialog,
                                height=200,
                                key=f"{model_id}_dialog_{j}",
                                disabled=True
                            )
            else:
                st.info(f"No results for {model_id} yet") 