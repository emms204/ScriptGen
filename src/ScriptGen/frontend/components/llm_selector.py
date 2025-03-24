"""
LLM selection component for Streamlit frontend.

This component provides a user interface for selecting and configuring LLMs.
"""

import streamlit as st
from typing import Dict, List, Any, Optional, Callable

from ..utils.session import get_state, set_state


def llm_selector(on_change: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Component for LLM selection and configuration.
    
    Args:
        on_change: Optional callback function to call when LLM config changes
        
    Returns:
        Dictionary with selected LLM configuration
    """
    # Get current LLM config from session state
    current_config = get_state("llm_config", {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000
    })
    
    # Create container for LLM selection
    with st.container():
        st.markdown("### LLM Selection")
        
        # Provider selection
        providers = ["openai", "anthropic", "local"]
        provider_names = {"openai": "OpenAI", "anthropic": "Anthropic", "local": "Local"}
        
        provider = st.selectbox(
            "Provider",
            options=providers,
            format_func=lambda x: provider_names.get(x, x),
            index=providers.index(current_config["provider"]) if current_config["provider"] in providers else 0
        )
        
        # Model selection based on provider
        models = {
            "openai": ["gpt-4", "gpt-3.5-turbo"],
            "anthropic": ["claude-2", "claude-instant"],
            "local": ["llama-2", "mistral"]
        }
        
        model_names = {
            "gpt-4": "GPT-4",
            "gpt-3.5-turbo": "GPT-3.5 Turbo",
            "claude-2": "Claude 2",
            "claude-instant": "Claude Instant",
            "llama-2": "Llama 2",
            "mistral": "Mistral"
        }
        
        available_models = models.get(provider, ["none"])
        
        # Try to keep the current model if possible, otherwise default to first model
        default_index = 0
        if current_config["model"] in available_models:
            default_index = available_models.index(current_config["model"])
        
        model = st.selectbox(
            "Model",
            options=available_models,
            format_func=lambda x: model_names.get(x, x),
            index=default_index
        )
        
        # Parameter configuration
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=current_config.get("temperature", 0.7),
                step=0.05,
                help="Higher values make output more random, lower values more deterministic"
            )
        
        with col2:
            max_tokens = st.slider(
                "Max Tokens",
                min_value=100,
                max_value=4000,
                value=current_config.get("max_tokens", 1000),
                step=100,
                help="Maximum number of tokens to generate in response"
            )
        
        # Optional: Sample output button
        show_sample = st.checkbox("Show sample output", value=False)
        if show_sample:
            st.caption("Sample response would be loaded here in a real implementation")
        
        # Create new config
        new_config = {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Check if config changed
        config_changed = new_config != current_config
        
        # Update session state if config changed
        if config_changed:
            set_state("llm_config", new_config)
            if on_change:
                on_change(new_config)
        
        # Return the current config
        return new_config


def llm_config_display(config: Dict[str, Any]) -> None:
    """
    Display current LLM configuration.
    
    Args:
        config: LLM configuration to display
    """
    provider_names = {"openai": "OpenAI", "anthropic": "Anthropic", "local": "Local"}
    model_names = {
        "gpt-4": "GPT-4",
        "gpt-3.5-turbo": "GPT-3.5 Turbo",
        "claude-2": "Claude 2",
        "claude-instant": "Claude Instant",
        "llama-2": "Llama 2",
        "mistral": "Mistral"
    }
    
    provider = provider_names.get(config.get("provider", ""), config.get("provider", "Unknown"))
    model = model_names.get(config.get("model", ""), config.get("model", "Unknown"))
    
    st.markdown(f"""
    **Current LLM:**  
    {provider} / {model}  
    Temperature: {config.get('temperature', 0.7)}  
    Max Tokens: {config.get('max_tokens', 1000)}
    """)


def try_llm_button(on_click: Optional[Callable] = None) -> None:
    """
    Button to test the LLM with a sample prompt.
    
    Args:
        on_click: Function to call when button is clicked
    """
    if st.button("Try this LLM", key="try_llm"):
        with st.spinner("Generating sample..."):
            if on_click:
                on_click()
            else:
                # Default behavior
                st.info("Sample response would be generated here in a real implementation") 