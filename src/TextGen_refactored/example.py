#!/usr/bin/env python3
"""
Example script demonstrating how to use the refactored TextGen module.

This example shows how to:
1. Set up a language model provider
2. Create a story generator
3. Generate a complete story with a given logline
4. Render the story to a formatted output
"""

import os
import sys
from pathlib import Path
import logging

# Add the parent directory to sys.path to make imports work
sys.path.append(str(Path(__file__).parent.parent))

from TextGen_refactored import (
    OpenAIProvider,
    GeminiProvider, 
    AnthropicProvider,
    ProviderFactory,
    StoryGenerator,
    EnhancedStoryGenerator,
    render_story
)
from TextGen_refactored.config import ModelConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run a demo of the TextGen module"""
    # Get API keys from environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    
    # Check if we have at least one API key
    if not any([openai_api_key, anthropic_api_key, gemini_api_key]):
        logger.error("No API keys found in environment variables. Please set at least one of: "
                    "OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY")
        return
    
    # Create a provider factory
    factory = ProviderFactory()
    
    # Choose a provider based on available API keys
    provider = None
    model_config = None
    
    if openai_api_key:
        logger.info("Using OpenAI provider")
        model_config = ModelConfig(
            provider="openai",
            model="gpt-4",
            max_tokens=1000,
            temperature=0.7,
            api_key=openai_api_key
        )
        provider = factory.create_provider(model_config)
    elif anthropic_api_key:
        logger.info("Using Anthropic provider")
        model_config = ModelConfig(
            provider="anthropic",
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.7,
            api_key=anthropic_api_key
        )
        provider = factory.create_provider(model_config)
    elif gemini_api_key:
        logger.info("Using Gemini provider")
        model_config = ModelConfig(
            provider="gemini",
            model="gemini-1.5-pro",
            max_tokens=1000,
            temperature=0.7,
            api_key=gemini_api_key
        )
        provider = factory.create_provider(model_config)
        
    if not provider:
        logger.error("Failed to create a provider")
        return
    
    # Sample logline
    logline = ("A brilliant but eccentric mathematician discovers an equation "
              "that predicts future events, but struggles with the ethical "
              "implications when a corporation tries to weaponize the discovery.")
    
    # Create a story generator (use the enhanced version for toxicity filtering)
    generator = EnhancedStoryGenerator(provider)
    
    # Generate a story
    logger.info(f"Generating story with logline: {logline}")
    story = generator.generate_story(logline)
    
    # Render the story
    formatted_story = render_story(story)
    
    # Save to a file
    output_path = Path("generated_story.txt")
    with open(output_path, "w") as f:
        f.write(formatted_story)
    
    logger.info(f"Story generated and saved to {output_path.absolute()}")
    
    # Print a small preview
    print("\nStory preview:")
    print("==============")
    print(f"Title: {story.title.text}")
    print("\nCharacters:")
    for character in story.characters.characters[:2]:  # Just first 2 characters
        print(f"- {character.name}: {character.description[:100]}...")
    print("\nScene 1 Preview:")
    if story.scenes and story.scenes.scenes:
        first_scene = story.scenes.scenes[0]
        print(f"- {first_scene.title}")
        print(f"  {first_scene.content[:150]}...")
    
    print(f"\nFull story saved to {output_path.absolute()}")

if __name__ == "__main__":
    main() 