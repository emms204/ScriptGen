"""
Export utilities for the Streamlit interface.

This module provides functions to export stories in different formats.
"""
import json
from typing import Dict, Any, List
import csv
import io
from xml.dom import minidom

from TextGen_refactored.models import Story


def export_as_text(story: Story) -> str:
    """Export a story as plain text.
    
    Args:
        story: The story to export
        
    Returns:
        Story formatted as plain text
    """
    result = []
    
    # Add title and storyline
    if hasattr(story.title, 'title'):
        result.append(f"Title: {story.title.title}")
    else:
        result.append(f"Title: {story.title}")
    
    result.append(f"\nLogline: {story.storyline}\n")
    
    # Add character descriptions
    result.append("Characters:")
    for name, desc in story.character_descriptions.items():
        result.append(f"  {name} - {desc}")
    
    result.append("")
    
    # Add scenes and dialogs
    if story.scenes and story.scenes.scenes:
        for i, (scene, dialog) in enumerate(zip(story.scenes.scenes, story.dialogs)):
            result.append(f"Scene {i+1}: {scene.place} - {scene.plot_element}")
            result.append(f"  Place: {scene.place}")
            result.append(f"  Plot Element: {scene.plot_element}")
            result.append(f"  Beat: {scene.beat}")
            
            # Add place description if available
            if scene.place in story.place_descriptions:
                result.append("\n  Place Description:")
                result.append(f"  {story.place_descriptions[scene.place].description}")
            
            # Add dialog
            result.append("\n  Dialog:")
            for line in dialog.split("\n"):
                result.append(f"    {line}")
            
            result.append("")
    
    return "\n".join(result)


def story_to_dict(story: Story) -> Dict[str, Any]:
    """Convert a story to a dictionary.
    
    Args:
        story: The story to convert
        
    Returns:
        Story represented as a dictionary
    """
    # Convert title to string if needed
    if hasattr(story.title, 'title'):
        title = story.title.title
    else:
        title = str(story.title)
    
    # Convert scenes to list of dicts
    scenes = []
    if story.scenes and story.scenes.scenes:
        for scene in story.scenes.scenes:
            scenes.append({
                'place': scene.place,
                'plot_element': scene.plot_element,
                'beat': scene.beat
            })
    
    # Convert place descriptions to dict
    places = {}
    for name, place in story.place_descriptions.items():
        places[name] = place.description
    
    # Create the result dictionary
    result = {
        'title': title,
        'storyline': story.storyline,
        'characters': story.character_descriptions,
        'places': places,
        'scenes': scenes,
        'dialogs': story.dialogs
    }
    
    return result


def export_as_json(story: Story) -> str:
    """Export a story as JSON.
    
    Args:
        story: The story to export
        
    Returns:
        Story formatted as JSON
    """
    story_dict = story_to_dict(story)
    return json.dumps(story_dict, indent=2)


def export_as_csv(story: Story) -> str:
    """Export a story as CSV.
    
    Args:
        story: The story to export
        
    Returns:
        Story scenes and dialogs formatted as CSV
    """
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Scene', 'Place', 'Plot Element', 'Beat', 'Dialog'])
    
    # Write data
    if story.scenes and story.scenes.scenes:
        for i, (scene, dialog) in enumerate(zip(story.scenes.scenes, story.dialogs)):
            writer.writerow([
                i+1,
                scene.place,
                scene.plot_element,
                scene.beat,
                dialog.replace('\n', ' ')
            ])
    
    return output.getvalue()


def export_as_fountain(story: Story) -> str:
    """Export a story in Fountain format for screenplays.
    
    Args:
        story: The story to export
        
    Returns:
        Story formatted in Fountain markup
    """
    result = []
    
    # Title page
    if hasattr(story.title, 'title'):
        result.append(f"Title: {story.title.title}")
    else:
        result.append(f"Title: {story.title}")
    
    result.append("Credit: Written by")
    result.append("Author: DramaTron AI")
    result.append(f"Source: {story.storyline}")
    result.append("Draft date: " + " ")  # Add current date later
    result.append("Contact: ")
    result.append("")
    
    # Add scenes and dialogs
    if story.scenes and story.scenes.scenes:
        for i, (scene, dialog) in enumerate(zip(story.scenes.scenes, story.dialogs)):
            # Scene heading
            result.append(f"INT. {scene.place.upper()} - DAY")
            
            # Scene description/action
            result.append(f"{scene.beat}")
            result.append("")
            
            # Dialog
            for line in dialog.split("\n"):
                # Check if it's a character name followed by dialog
                if ":" in line:
                    parts = line.split(":", 1)
                    character = parts[0].strip().upper()
                    dialog_text = parts[1].strip()
                    
                    result.append(character)
                    result.append(dialog_text)
                    result.append("")
                else:
                    # Action/direction
                    result.append(line)
                    result.append("")
            
            result.append("")
    
    return "\n".join(result)


def export_as_xml(story: Story) -> str:
    """Export a story as XML.
    
    Args:
        story: The story to export
        
    Returns:
        Story formatted as XML
    """
    doc = minidom.getDOMImplementation().createDocument(None, "story", None)
    root = doc.documentElement
    
    # Add title and storyline
    title_elem = doc.createElement("title")
    if hasattr(story.title, 'title'):
        title_elem.appendChild(doc.createTextNode(story.title.title))
    else:
        title_elem.appendChild(doc.createTextNode(str(story.title)))
    root.appendChild(title_elem)
    
    storyline_elem = doc.createElement("storyline")
    storyline_elem.appendChild(doc.createTextNode(story.storyline))
    root.appendChild(storyline_elem)
    
    # Add characters
    characters_elem = doc.createElement("characters")
    for name, desc in story.character_descriptions.items():
        char_elem = doc.createElement("character")
        
        name_elem = doc.createElement("name")
        name_elem.appendChild(doc.createTextNode(name))
        char_elem.appendChild(name_elem)
        
        desc_elem = doc.createElement("description")
        desc_elem.appendChild(doc.createTextNode(desc))
        char_elem.appendChild(desc_elem)
        
        characters_elem.appendChild(char_elem)
    
    root.appendChild(characters_elem)
    
    # Add places
    places_elem = doc.createElement("places")
    for name, place in story.place_descriptions.items():
        place_elem = doc.createElement("place")
        
        name_elem = doc.createElement("name")
        name_elem.appendChild(doc.createTextNode(name))
        place_elem.appendChild(name_elem)
        
        desc_elem = doc.createElement("description")
        desc_elem.appendChild(doc.createTextNode(place.description))
        place_elem.appendChild(desc_elem)
        
        places_elem.appendChild(place_elem)
    
    root.appendChild(places_elem)
    
    # Add scenes and dialogs
    scenes_elem = doc.createElement("scenes")
    if story.scenes and story.scenes.scenes:
        for i, (scene, dialog) in enumerate(zip(story.scenes.scenes, story.dialogs)):
            scene_elem = doc.createElement("scene")
            scene_elem.setAttribute("id", str(i+1))
            
            place_elem = doc.createElement("place")
            place_elem.appendChild(doc.createTextNode(scene.place))
            scene_elem.appendChild(place_elem)
            
            plot_elem = doc.createElement("plot_element")
            plot_elem.appendChild(doc.createTextNode(scene.plot_element))
            scene_elem.appendChild(plot_elem)
            
            beat_elem = doc.createElement("beat")
            beat_elem.appendChild(doc.createTextNode(scene.beat))
            scene_elem.appendChild(beat_elem)
            
            dialog_elem = doc.createElement("dialog")
            dialog_elem.appendChild(doc.createTextNode(dialog))
            scene_elem.appendChild(dialog_elem)
            
            scenes_elem.appendChild(scene_elem)
    
    root.appendChild(scenes_elem)
    
    return doc.toprettyxml(indent="  ") 