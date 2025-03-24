#!/usr/bin/env python3
"""
WorkflowController Example Script

This example demonstrates the end-to-end script generation process using
the WorkflowController, which orchestrates all modules in the ScriptGen system.

It showcases:
1. Bible bootstrapping
2. Scene generation
3. Multi-scene script generation
4. LLM optimization

Usage:
    python -m src.ScriptGen.examples.workflow_example [--full] [--scenes=N] [--optimize]
"""

import os
import json
import argparse
import time
from pprint import pprint
from typing import Dict, List, Any, Optional

from ..bibles.bible_storage import BibleStorage
from ..llm.llm_wrapper import LLMWrapper
from ..orchestrator.workflow_controller import (
    WorkflowController,
    GenerationConfig,
    GenerationScope
)
from ..orchestrator.optimizer import CacheConfig


def save_output(output_data: Dict[str, Any], filename: str) -> None:
    """
    Save output data to a JSON file.
    
    Args:
        output_data: Data to save
        filename: Output filename
    """
    try:
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Output saved to {filename}")
    except Exception as e:
        print(f"Error saving output: {e}")


def print_scene_summary(scene: Dict[str, Any]) -> None:
    """
    Print a summary of a scene.
    
    Args:
        scene: Scene data
    """
    scene_id = scene.get('scene_id', 'Unknown')
    print(f"\n--- Scene: {scene_id} ---")
    
    # Print outline
    outline = scene.get('outline', {})
    print(f"Location: {outline.get('location', 'Unknown')}")
    print(f"Description: {outline.get('description', 'No description')}")
    
    # Print characters
    character_dialogue = scene.get('character_dialogue', [])
    if character_dialogue:
        print("\nCharacters:")
        for char in character_dialogue:
            print(f"- {char.get('character', 'Unknown')}")
    
    # Print first few lines of integrated scene
    integrated_scene = scene.get('integrated_scene', '')
    if integrated_scene:
        print("\nScene preview:")
        lines = integrated_scene.split('\n')
        preview_lines = lines[:5]
        print('\n'.join(preview_lines))
        if len(lines) > 5:
            print("...")
    
    print(f"--- End Scene: {scene_id} ---\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="WorkflowController Example")
    parser.add_argument('log_line', nargs='?', 
                      default="A detective with amnesia must solve a murder that he might have committed.",
                      help="Log line describing the script")
    parser.add_argument('--full', action='store_true',
                      help="Generate full script with act structure")
    parser.add_argument('--scenes', type=int, default=3,
                      help="Number of scenes to generate")
    parser.add_argument('--optimize', action='store_true',
                      help="Use LLM optimization")
    parser.add_argument('--characters', type=int, default=2,
                      help="Number of character bibles to create")
    parser.add_argument('--output', help="Output filename")
    args = parser.parse_args()
    
    print(f"ScriptGen Workflow Example")
    print("=" * 50)
    print(f"Log Line: {args.log_line}")
    
    # Create components
    bible_storage = BibleStorage(db_path=':memory:')
    llm_wrapper = LLMWrapper()
    
    # Configure optimization if requested
    cache_config = None
    if args.optimize:
        print("LLM optimization enabled")
        cache_config = CacheConfig(
            enable_cache=True,
            cache_dir=os.path.join(os.path.dirname(__file__), 'cache'),
            max_cache_entries=100
        )
    
    # Determine generation scope
    if args.full:
        scope = GenerationScope.FULL_SCRIPT
        print(f"Generating full script with {args.scenes} scenes")
    elif args.scenes > 1:
        scope = GenerationScope.MULTI_SCENE
        print(f"Generating multi-scene script with {args.scenes} scenes")
    else:
        scope = GenerationScope.SINGLE_SCENE
        print("Generating single scene")
    
    # Create generation config
    config = GenerationConfig(
        scope=scope,
        character_count=args.characters,
        scene_count=args.scenes,
        use_existing_bibles=False,
        auto_rewrite=True,
        rewrite_iterations=1,
        parallel_generation=True,
        optimize_llm_calls=args.optimize,
        cache_config=cache_config
    )
    
    # Create workflow controller
    controller = WorkflowController(
        bible_storage=bible_storage,
        llm_wrapper=llm_wrapper
    )
    
    # Start timing
    start_time = time.time()
    
    try:
        # Generate script
        print("\nGenerating script...")
        result = controller.generate_script(args.log_line, config)
        
        # Print stats
        stats = result.get('stats', {})
        print("\n--- Generation Stats ---")
        print(f"Scenes generated: {stats.get('scenes_generated', 0)}")
        print(f"LLM calls: {stats.get('llm_calls', 0)}")
        print(f"Duration: {stats.get('duration', 0):.2f} seconds")
        
        # Print scene summaries
        scenes = result.get('scenes', [])
        for scene in scenes:
            print_scene_summary(scene)
        
        # Save output if requested
        if args.output:
            save_output(result, args.output)
        
    except Exception as e:
        print(f"Error: {e}")
    
    # End timing
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main() 