#!/usr/bin/env python3
"""
Example script demonstrating the Iterative Rewrite Engine.

This script showcases:
1. Scene critique using the Dramaturge Agent
2. Scene rewriting based on user instructions
3. Bible change propagation across dependent scenes
"""

import os
import sys
import json
import argparse
from typing import Optional, Dict, Any
from enum import Enum

# Add parent directory to path to allow importing from ScriptGen
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from TextGen.ScriptGen.bibles.bible_storage import BibleStorage
from TextGen.ScriptGen.llm.llm_wrapper import LLMWrapper
from TextGen.ScriptGen.rewrite.rewrite_controller import (
    RewriteController, ChangeType, ChangeRequest, SceneCritique
)


def create_sample_data(bible_storage: BibleStorage) -> Dict[str, Any]:
    """
    Create sample bible entries and scenes for demonstration.
    
    Args:
        bible_storage: BibleStorage instance
        
    Returns:
        Dictionary with created bible IDs and scene data
    """
    print("Creating sample bible entries and scenes...")
    
    # Character: Theo
    character = {
        'name': 'Theo',
        'backstory': 'A talented singer whose career was destroyed in a fire at a nightclub.',
        'goals': 'Seek revenge against the club owner who locked the fire exits',
        'fears': 'Being forgotten, fire',
        'traits': ['determined', 'bitter', 'talented'],
        'relationships': {
            'Marcus': 'Former mentor, now the target of revenge',
            'Lila': 'Friend who survived the fire but was injured'
        }
    }
    character_id = bible_storage.create_entry('character', character)
    
    # Plot: Revenge Plot
    plot = {
        'title': 'Vengeance Song',
        'theme': 'Revenge vs. Redemption',
        'act_structure': '3-act',
        'main_conflict': 'Theo\'s desire for revenge vs. opportunity for healing',
        'scenes': [
            {
                'scene_id': 'act1_scene2',
                'description': 'Theo confronts Marcus at his new club',
                'prior_state': 'Theo has just discovered Marcus opened a new venue',
                'next_state': 'Theo begins planning sabotage'
            }
        ]
    }
    plot_id = bible_storage.create_entry('plot', plot)
    
    # Sample scene
    scene = {
        'scene_id': 'act1_scene2',
        'title': 'The Confrontation',
        'character_ids': [character_id],
        'plot_id': plot_id,
        'outline': {
            'location': 'Upscale nightclub - "The Phoenix"',
            'time': 'Night',
            'description': 'Theo confronts Marcus at his new club',
            'objectives': ['Establish Theo\'s bitterness', 'Reveal Marcus\'s indifference', 'Set up future conflict']
        },
        'integrated_scene': """INT. THE PHOENIX NIGHTCLUB - NIGHT

Sleek, modern decor with fire-themed accents. The club is busy, patrons dancing and drinking. At the bar, MARCUS (50s, confident, well-dressed) oversees his staff.

THEO (30s, scarred face, intense) enters and scans the room. Spotting Marcus, he pushes through the crowd.

THEO
(coldly)
Hello, Marcus.

Marcus turns, momentarily frozen in recognition.

MARCUS
(recovering)
Theo. Didn't expect to see you... alive.

THEO
(bitter)
Sorry to disappoint. Nice place. Better fire exits in this one?

MARCUS
(dismissive)
That was years ago. Tragic accident. I was cleared of all charges.

THEO
(leaning in)
We both know what happened. You locked those doors to prevent walkouts.

MARCUS
(threatening)
Careful, Theo. The past is past. You should move on... while you can.

THEO
(with quiet intensity)
I haven't sung a note since that night. The smoke damaged my voice permanently.

MARCUS
(shrugging)
That's show business. Some careers burn bright, others... just burn.

Theo's hands ball into fists. A SECURITY GUARD notices and approaches.

MARCUS
(smirking)
I think it's time for you to leave.

THEO
(as security arrives)
This isn't over.

Security escorts Theo toward the exit. Theo glances back, seeing Marcus already turning away, unbothered.
"""
    }
    
    print("Sample data created successfully:")
    print(f"- Character 'Theo' (ID: {character_id})")
    print(f"- Plot 'Vengeance Song' (ID: {plot_id})")
    print(f"- Scene 'The Confrontation'")
    
    return {
        'character_id': character_id,
        'plot_id': plot_id,
        'scene': scene
    }


def demonstrate_critique(controller: RewriteController, scene: Dict[str, Any]) -> SceneCritique:
    """
    Demonstrate scene critique using the Dramaturge Agent.
    
    Args:
        controller: RewriteController instance
        scene: Scene data
    
    Returns:
        Critique results
    """
    print("\n" + "=" * 80)
    print("DEMONSTRATION: SCENE CRITIQUE")
    print("=" * 80)
    
    print("Analyzing scene for pacing, tension, subtext, and other dramatic elements...")
    critique = controller.critique_scene(scene['scene_id'], scene)
    
    print("\nCRITIQUE RESULTS:")
    print(f"\nOverall Assessment:\n{critique.overall_assessment}")
    
    print("\nPacing Issues:")
    for issue in critique.pacing.get('issues', []):
        print(f"- {issue}")
    
    print("\nTension Suggestions:")
    for suggestion in critique.tension.get('suggestions', []):
        print(f"- {suggestion}")
        
    print("\nSubtext Analysis:")
    for issue in critique.subtext.get('issues', []):
        print(f"- {issue}")
    
    print("\nSuggested Rewrites:")
    for rewrite in critique.suggested_rewrites:
        print(f"- {rewrite}")
    
    return critique


def demonstrate_rewrite(
    controller: RewriteController, 
    scene: Dict[str, Any],
    user_input: str
) -> Dict[str, Any]:
    """
    Demonstrate scene rewriting based on user instructions.
    
    Args:
        controller: RewriteController instance
        scene: Scene data
        user_input: User instructions for rewriting
    
    Returns:
        Rewrite results
    """
    print("\n" + "=" * 80)
    print("DEMONSTRATION: SCENE REWRITE")
    print("=" * 80)
    
    print(f"Rewriting scene with instructions: \"{user_input}\"")
    rewrite_results = controller.rewrite_scene(
        scene_id=scene['scene_id'],
        scene_content=scene,
        user_input=user_input
    )
    
    print("\nREWRITE COMPLETED")
    print("\nOriginal Scene (excerpt):")
    original_lines = scene['integrated_scene'].split('\n')[:10]
    print('\n'.join(original_lines) + '\n...')
    
    print("\nRewritten Scene (excerpt):")
    new_lines = rewrite_results['new_content']['integrated_scene'].split('\n')[:10]
    print('\n'.join(new_lines) + '\n...')
    
    print("\nDiff (excerpt):")
    diff_lines = rewrite_results['diff'].split('\n')[:20]
    print('\n'.join(diff_lines) + '\n...')
    
    return rewrite_results


def demonstrate_change_propagation(
    controller: RewriteController,
    bible_storage: BibleStorage,
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Demonstrate bible change propagation across dependent scenes.
    
    Args:
        controller: RewriteController instance
        bible_storage: BibleStorage instance
        data: Sample data including bible IDs and scene
    
    Returns:
        Propagation results
    """
    print("\n" + "=" * 80)
    print("DEMONSTRATION: CHANGE PROPAGATION")
    print("=" * 80)
    
    # Get the original character bible
    character_id = data['character_id']
    character = bible_storage.get_entry(character_id)
    character_name = character['content']['name']
    
    # Create a change request to rename the character
    new_name = "Teddy"
    print(f"Changing character name from '{character_name}' to '{new_name}'")
    
    change_request = ChangeRequest(
        bible_id=character_id,
        bible_type='character',
        change_type=ChangeType.NAME,
        field_path='content.name',
        old_value=character_name,
        new_value=new_name,
        description=f"Changing character name from {character_name} to {new_name}",
        propagate=True
    )
    
    # Generate change plan
    print("\nGenerating change propagation plan...")
    change_plan = controller.propagate_change(change_request)
    
    print(f"\nFound {len(change_plan.affected_scenes)} affected scenes")
    
    # Show preview of changes
    for scene in change_plan.affected_scenes:
        print(f"\nAffected Scene: {scene.title}")
        print("Change Preview (excerpt):")
        if scene.diff:
            diff_lines = scene.diff.split('\n')[:15]
            print('\n'.join(diff_lines) + '\n...')
    
    # Apply changes
    print("\nApplying changes...")
    scene_ids = [scene.scene_id for scene in change_plan.affected_scenes]
    results = controller.apply_propagation(change_plan, scene_ids)
    
    print("\nChange Application Results:")
    print(f"- Bible updated: {results['bible_updated']}")
    print(f"- Scenes updated: {len(results['scenes_updated'])}")
    
    # Verify the character name was updated
    updated_character = bible_storage.get_entry(character_id)
    print(f"\nCharacter name is now: {updated_character['content']['name']}")
    
    return results


def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(description="Example script for Iterative Rewrite Engine")
    parser.add_argument('--config', help="Path to LLM config file")
    args = parser.parse_args()
    
    print("Iterative Rewrite Engine Example")
    print("=" * 80)
    
    # Initialize components with in-memory database
    bible_storage = BibleStorage(db_path=':memory:')
    llm_wrapper = LLMWrapper(config_path=args.config)
    controller = RewriteController(
        bible_storage=bible_storage,
        llm_wrapper=llm_wrapper
    )
    
    # Create sample data
    data = create_sample_data(bible_storage)
    
    # Demo 1: Scene Critique
    critique = demonstrate_critique(controller, data['scene'])
    
    # Demo 2: Scene Rewrite
    rewrite_results = demonstrate_rewrite(
        controller, 
        data['scene'],
        "Make Theo's character more sympathetic and add more subtext to the dialogue"
    )
    
    # Update the scene with the rewritten content for the next demo
    data['scene'] = rewrite_results['new_content']
    
    # Demo 3: Change Propagation
    propagation_results = demonstrate_change_propagation(
        controller,
        bible_storage,
        data
    )
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main() 