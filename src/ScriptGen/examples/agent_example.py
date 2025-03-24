#!/usr/bin/env python3
"""
Example script demonstrating the Agent Framework for script generation.

This script uses the Character, Director, and Dramaturge agents to collaboratively
generate a script scene from character and plot bible entries.
"""

import os
import sys
import json
import argparse
from typing import Optional, List, Dict, Any

# Add parent directory to path to allow importing from ScriptGen
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ScriptGen.bibles.bible_storage import BibleStorage
from ScriptGen.llm.llm_wrapper import LLMWrapper
from ScriptGen.agents.agent_framework import run_agents, batch_character_agent_call


def create_sample_bibles(bible_storage: BibleStorage) -> Dict[str, Any]:
    """
    Create sample bible entries for demonstration.
    
    Args:
        bible_storage: BibleStorage instance
        
    Returns:
        Dictionary with created bible IDs
    """
    print("Creating sample bible entries...")
    
    # Character 1: Detective
    detective = {
        'name': 'Detective Alex Morgan',
        'backstory': 'A veteran detective suffering from amnesia after a head injury. Haunted by the possibility that they might have committed a crime they cannot remember.',
        'goals': 'Solve the murder case and recover lost memories',
        'fears': 'Discovering they are responsible for the murder',
        'traits': ['determined', 'analytical', 'haunted'],
        'relationships': {
            'Dr. Sarah Chen': 'Professional relationship, somewhat distrustful of her motives',
            'Captain Reynolds': 'Boss and former friend, now wary of Alex\'s condition'
        }
    }
    detective_id = bible_storage.create_entry('character', detective)
    
    # Character 2: Forensic Psychologist
    psychologist = {
        'name': 'Dr. Sarah Chen',
        'backstory': 'A brilliant forensic psychologist assigned to help Detective Morgan. Has her own theories about the case that she hasn\'t fully shared.',
        'goals': 'Uncover the truth and study the detective\'s psychological state',
        'fears': 'Being wrong about the detective\'s innocence',
        'traits': ['intelligent', 'observant', 'compassionate'],
        'relationships': {
            'Detective Alex Morgan': 'Professional interest mixed with genuine concern',
            'The Victim': 'Knew them peripherally through professional circles'
        }
    }
    psychologist_id = bible_storage.create_entry('character', psychologist)
    
    # Plot: Crime Scene Investigation
    plot = {
        'scene_id': 'act1_scene2',
        'act': 1,
        'description': 'Detective Morgan and Dr. Chen investigate the crime scene, where Morgan experiences troubling memory flashes.',
        'prior_state': 'Morgan has just been assigned to a murder case with peculiar similarities to their memory gaps.',
        'next_state': 'Morgan finds evidence that suggests they were at the crime scene before.',
        'key_elements': [
            'Memory flash triggered by specific evidence',
            'Tension between detective and psychologist',
            'Discovery of a personal item that connects the detective to the scene'
        ]
    }
    plot_id = bible_storage.create_entry('plot', plot)
    
    # Setting: Crime Scene
    setting = {
        'location_id': 'abandoned_apartment',
        'name': 'Abandoned Apartment 5B',
        'description': 'A dilapidated apartment in a condemned building. Water damage on the walls, minimal furniture. Signs of struggle.',
        'constraints': 'Single room, one entrance/exit, broken window, poor lighting',
        'mood': 'Oppressive, tense, shadowy'
    }
    setting_id = bible_storage.create_entry('setting', setting)
    
    print("Sample bible entries created successfully.")
    
    return {
        'detective_id': detective_id,
        'psychologist_id': psychologist_id,
        'plot_id': plot_id,
        'setting_id': setting_id
    }


def generate_scene_with_agents(
    bible_storage: BibleStorage,
    llm_wrapper: LLMWrapper,
    bible_ids: Dict[str, int],
    use_batch: bool = False,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a scene using the agent framework.
    
    Args:
        bible_storage: BibleStorage instance
        llm_wrapper: LLMWrapper instance
        bible_ids: Dictionary with bible entry IDs
        use_batch: Whether to use batched character agent calls
        output_path: Optional path to save the results
        
    Returns:
        Generated scene data
    """
    print("\nGenerating scene with agents...")
    
    # Retrieve bible entries
    detective = bible_storage.get_entry(bible_ids['detective_id'])
    psychologist = bible_storage.get_entry(bible_ids['psychologist_id'])
    plot = bible_storage.get_entry(bible_ids['plot_id'])
    setting = bible_storage.get_entry(bible_ids['setting_id'])
    
    # Create scene context
    scene_context = {
        'scene_id': plot['content']['scene_id'],
        'location': setting['content']['name'],
        'description': plot['content']['description'],
        'objectives': [
            'Reveal the detective\'s memory flash',
            'Establish tension between the characters',
            'Discover a personal item connecting the detective to the scene'
        ],
        'mood': setting['content']['mood'],
        'time': 'Night',
        'setting_details': setting['content']['description']
    }
    
    if use_batch:
        print("Using batched character agent calls...")
        # Use batched character agent calls
        characters = [detective, psychologist]
        batch_result = batch_character_agent_call(
            characters=characters,
            scene_context=json.dumps(scene_context),
            llm_wrapper=llm_wrapper
        )
        
        # Extract character outputs
        character_outputs = batch_result['character_outputs']
        
        # Run director and dramaturge agents
        director_agent = DirectorAgent(llm_wrapper)
        integrated_scene = director_agent.integrate_scene(
            character_outputs=character_outputs,
            plot_bible=plot,
            scene_outline=scene_context
        )
        
        dramaturge_agent = DramaturgeAgent(llm_wrapper)
        scene_critique = dramaturge_agent.critique_scene(
            scene=integrated_scene['integrated_scene'],
            plot_bible=plot
        )
        
        # Combine results
        result = {
            'scene_id': scene_context['scene_id'],
            'outline': scene_context,
            'character_dialogue': character_outputs,
            'integrated_scene': integrated_scene['integrated_scene'],
            'critique': scene_critique['critique']
        }
    else:
        print("Using standard run_agents function...")
        # Use standard run_agents function
        result = run_agents(
            scene_id=plot['content']['scene_id'],
            characters=[detective, psychologist],
            plot_bible=plot,
            scene_context=scene_context,
            bible_storage=bible_storage,
            llm_wrapper=llm_wrapper,
            parallel=True
        )
    
    # Save results if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    return result


def display_results(result: Dict[str, Any]) -> None:
    """
    Display the generated scene results.
    
    Args:
        result: The generated scene data
    """
    print("\n" + "=" * 80)
    print("SCENE OUTLINE:")
    print("=" * 80)
    print(json.dumps(result['outline'], indent=2))
    
    print("\n" + "=" * 80)
    print("CHARACTER DIALOGUE:")
    print("=" * 80)
    for char_output in result['character_dialogue']:
        print(f"\n{char_output['character']}:")
        for line in char_output['dialogue_lines']:
            action = line.get('action', '')
            text = line.get('text', '')
            print(f"  {action} {text}")
    
    print("\n" + "=" * 80)
    print("INTEGRATED SCENE:")
    print("=" * 80)
    print(result['integrated_scene'])
    
    print("\n" + "=" * 80)
    print("DRAMATURGE CRITIQUE:")
    print("=" * 80)
    print(result['critique'])


def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(description="Example script for Agent Framework")
    parser.add_argument('--config', help="Path to LLM config file")
    parser.add_argument('--output', help="Path to save output JSON")
    parser.add_argument('--batch', action='store_true', help="Use batched character agent calls")
    args = parser.parse_args()
    
    print("Agent Framework Example")
    print("=" * 80)
    
    # Initialize components
    bible_storage = BibleStorage(db_path=':memory:')
    llm_wrapper = LLMWrapper(config_path=args.config)
    
    # Create sample bibles
    bible_ids = create_sample_bibles(bible_storage)
    
    # Generate scene with agents
    from ScriptGen.agents.agent_framework import DirectorAgent, DramaturgeAgent
    result = generate_scene_with_agents(
        bible_storage=bible_storage,
        llm_wrapper=llm_wrapper,
        bible_ids=bible_ids,
        use_batch=args.batch,
        output_path=args.output
    )
    
    # Display results
    display_results(result)


if __name__ == "__main__":
    main() 