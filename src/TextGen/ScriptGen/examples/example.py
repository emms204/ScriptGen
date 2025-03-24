#!/usr/bin/env python3
"""
Example script showing how to use the ScriptGen system to generate script scenes.
"""

import os
import json
import sys
from typing import Optional

# Add parent directory to path to allow importing from ScriptGen
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from TextGen.ScriptGen.generators.script_generator import create_script_generator

def generate_script(log_line: str, config_path: Optional[str] = None):
    """
    Generate a script scene from a log line.
    
    Args:
        log_line: The log line to use as input
        config_path: Optional path to a config file with LLM settings
    """
    # Create script generator
    script_gen = create_script_generator(config_path)
    
    print(f"Generating script from log line: '{log_line}'")
    print("This may take a few moments...\n")
    
    # Generate script
    result = script_gen.generate_script_from_logline(log_line)
    
    # Print bible entries
    print("=" * 80)
    print("GENERATED BIBLE ENTRIES:")
    print("=" * 80)
    
    # Characters
    print("\nCHARACTERS:")
    for character in result['bible_entries'].get('characters', []):
        print(f"- {character['name']}")
        print(f"  Backstory: {character['backstory']}")
        print(f"  Goals: {character['goals']}")
        print(f"  Fears: {character['fears']}")
        
        if character.get('relationships'):
            print("  Relationships:")
            for other_char, relationship in character['relationships'].items():
                print(f"    - {other_char}: {relationship}")
        print()
    
    # Plot
    print("\nPLOT:")
    for plot in result['bible_entries'].get('plot', []):
        print(f"- Scene {plot['scene_id']} (Act {plot['act']})")
        print(f"  Description: {plot['description']}")
        print(f"  Prior state: {plot['prior_state']}")
        print(f"  Next state: {plot['next_state']}")
        print()
    
    # Settings
    print("\nSETTINGS:")
    for setting in result['bible_entries'].get('settings', []):
        print(f"- {setting['name']} ({setting['location_id']})")
        print(f"  Description: {setting['description']}")
        print(f"  Constraints: {setting['constraints']}")
        print()
    
    # Themes
    print("\nTHEMES:")
    for theme in result['bible_entries'].get('themes', []):
        print(f"- {theme['name']} ({theme['theme_id']})")
        print(f"  Description: {theme['description']}")
        print(f"  Examples: {', '.join(theme['examples'])}")
        print()
    
    # Print scene outline
    print("=" * 80)
    print("SCENE OUTLINE:")
    print("=" * 80)
    print(result['scene_outline']['text'])
    print()
    
    # Print dialogue
    print("=" * 80)
    print("DIALOGUE:")
    print("=" * 80)
    for line in result['dialogue']:
        print(f"{line['character']}: {line['text']}")
    print()
    
    # Print final script
    print("=" * 80)
    print("FINAL SCRIPT:")
    print("=" * 80)
    print(result['script'])
    
    # Save result to file
    with open('generated_script.json', 'w') as f:
        # Convert to JSON-serializable format (strip non-serializable elements)
        json_result = {
            'log_line': result['log_line'],
            'bible_entries': {
                'characters': result['bible_entries'].get('characters', []),
                'plot': result['bible_entries'].get('plot', []),
                'settings': result['bible_entries'].get('settings', []),
                'themes': result['bible_entries'].get('themes', [])
            },
            'scene_outline': {
                'scene_id': result['scene_outline']['scene_id'],
                'setting_id': result['scene_outline']['setting_id'],
                'text': result['scene_outline']['text'],
                'characters': result['scene_outline']['characters']
            },
            'dialogue': result['dialogue'],
            'script': result['script']
        }
        json.dump(json_result, f, indent=2)
    
    print(f"\nScript data saved to 'generated_script.json'")


if __name__ == "__main__":
    # Get log line from command line or use default
    if len(sys.argv) > 1:
        log_line = sys.argv[1]
    else:
        log_line = "Theo, a singer, seeks revenge after a fire destroys his career."
    
    # Get optional config path
    config_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    generate_script(log_line, config_path) 