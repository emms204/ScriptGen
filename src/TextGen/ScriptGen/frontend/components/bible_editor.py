"""
Bible editor component for Streamlit frontend.

This component provides a user interface for editing bible entries.
"""

import streamlit as st
from typing import Dict, List, Any, Optional, Callable

from ..utils.session import get_state, set_state, add_bible_change


def bible_editor(
    bible_type: str,
    bible_id: Optional[int] = None,
    initial_data: Optional[Dict[str, Any]] = None,
    on_save: Optional[Callable[[str, int, Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Editor component for bible entries.
    
    Args:
        bible_type: Type of bible entry (character, plot, setting, theme)
        bible_id: ID of the bible entry (None for new entries)
        initial_data: Initial data for the form
        on_save: Callback function to call when saving
        
    Returns:
        Dictionary with form data and save status
    """
    # Initialize result
    result = {
        "saved": False,
        "data": {},
        "canceled": False
    }
    
    # Standardize bible_type
    bible_type = bible_type.lower()
    
    # Prepare form key
    form_key = f"bible_editor_{bible_type}_{bible_id or 'new'}"
    
    # Define human-readable title
    type_titles = {
        "character": "Character",
        "plot": "Plot",
        "setting": "Setting",
        "theme": "Theme"
    }
    
    title = type_titles.get(bible_type, bible_type.capitalize())
    
    # Create the form
    with st.form(key=form_key):
        st.subheader(f"{'Edit' if bible_id else 'Create'} {title} Bible")
        
        # Get form fields based on bible type
        if bible_type == "character":
            result["data"] = character_bible_form(initial_data or {})
        elif bible_type == "plot":
            result["data"] = plot_bible_form(initial_data or {})
        elif bible_type == "setting":
            result["data"] = setting_bible_form(initial_data or {})
        elif bible_type == "theme":
            result["data"] = theme_bible_form(initial_data or {})
        else:
            st.warning(f"Unknown bible type: {bible_type}")
            result["data"] = generic_bible_form(initial_data or {})
        
        # Form buttons
        col1, col2 = st.columns(2)
        
        with col1:
            save_button = st.form_submit_button("Save")
        
        with col2:
            cancel_button = st.form_submit_button("Cancel")
        
        # Handle form submission
        if save_button:
            result["saved"] = True
            
            # Call save callback if provided
            if on_save:
                on_save(bible_type, bible_id, result["data"])
            
            # If editing an existing entry, track changes for propagation
            if bible_id is not None and initial_data:
                track_bible_changes(bible_id, bible_type, initial_data, result["data"])
        
        if cancel_button:
            result["canceled"] = True
    
    return result


def character_bible_form(initial_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Form for character bible entries.
    
    Args:
        initial_data: Initial form data
        
    Returns:
        Dictionary with form data
    """
    data = {}
    
    # Basic info
    data["name"] = st.text_input(
        "Name",
        value=initial_data.get("name", ""),
        help="Character's full name"
    )
    
    # Background
    data["backstory"] = st.text_area(
        "Backstory",
        value=initial_data.get("backstory", ""),
        help="Character's history and background"
    )
    
    # Motivations
    col1, col2 = st.columns(2)
    
    with col1:
        data["goals"] = st.text_area(
            "Goals",
            value=initial_data.get("goals", ""),
            help="What the character wants to achieve"
        )
    
    with col2:
        data["fears"] = st.text_area(
            "Fears",
            value=initial_data.get("fears", ""),
            help="What the character is afraid of"
        )
    
    # Traits
    traits_input = st.text_input(
        "Traits",
        value=", ".join(initial_data.get("traits", [])) if isinstance(initial_data.get("traits", []), list) else initial_data.get("traits", ""),
        help="Comma-separated list of character traits"
    )
    
    # Parse traits to list
    if traits_input:
        data["traits"] = [trait.strip() for trait in traits_input.split(",") if trait.strip()]
    else:
        data["traits"] = []
    
    # Relationships
    st.subheader("Relationships")
    
    # Get existing relationships
    relationships = initial_data.get("relationships", {})
    
    # Create a container for relationships
    data["relationships"] = {}
    
    # Display existing relationships
    for i, (person, relationship) in enumerate(relationships.items()):
        cols = st.columns([3, 6, 1])
        
        with cols[0]:
            new_person = st.text_input(
                "Person",
                value=person,
                key=f"rel_person_{i}"
            )
        
        with cols[1]:
            new_relationship = st.text_input(
                "Relationship",
                value=relationship,
                key=f"rel_desc_{i}"
            )
        
        # Only add if both fields are filled
        if new_person and new_relationship:
            data["relationships"][new_person] = new_relationship
    
    # Add new relationship option
    st.caption("Add new relationship:")
    cols = st.columns([3, 6, 1])
    
    with cols[0]:
        new_person = st.text_input(
            "Person",
            value="",
            key="rel_person_new"
        )
    
    with cols[1]:
        new_relationship = st.text_input(
            "Relationship",
            value="",
            key="rel_desc_new"
        )
    
    # Add to data if both fields are filled
    if new_person and new_relationship:
        data["relationships"][new_person] = new_relationship
    
    return data


def plot_bible_form(initial_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Form for plot bible entries.
    
    Args:
        initial_data: Initial form data
        
    Returns:
        Dictionary with form data
    """
    data = {}
    
    # Basic info
    data["title"] = st.text_input(
        "Title",
        value=initial_data.get("title", ""),
        help="Title of the plot"
    )
    
    data["theme"] = st.text_input(
        "Theme",
        value=initial_data.get("theme", ""),
        help="Central theme of the plot"
    )
    
    # Structure
    data["act_structure"] = st.selectbox(
        "Act Structure",
        options=["3-act", "4-act", "5-act", "Hero's Journey", "Other"],
        index=["3-act", "4-act", "5-act", "Hero's Journey", "Other"].index(initial_data.get("act_structure", "3-act")) if initial_data.get("act_structure") in ["3-act", "4-act", "5-act", "Hero's Journey", "Other"] else 0,
        help="Structural organization of the plot"
    )
    
    # Conflict
    data["main_conflict"] = st.text_area(
        "Main Conflict",
        value=initial_data.get("main_conflict", ""),
        help="Central conflict driving the plot"
    )
    
    # Scenes (simplified for form)
    st.subheader("Key Scenes")
    
    # Get existing scenes
    scenes = initial_data.get("scenes", [])
    
    # Create container for scenes
    data["scenes"] = []
    
    # Display existing scenes
    for i, scene in enumerate(scenes):
        with st.expander(f"Scene {i+1}: {scene.get('description', '')[0:30]}..."):
            scene_data = {}
            
            scene_data["scene_id"] = st.text_input(
                "Scene ID",
                value=scene.get("scene_id", f"scene_{i+1}"),
                key=f"scene_id_{i}"
            )
            
            scene_data["description"] = st.text_area(
                "Description",
                value=scene.get("description", ""),
                key=f"scene_desc_{i}"
            )
            
            scene_data["prior_state"] = st.text_input(
                "Prior State",
                value=scene.get("prior_state", ""),
                key=f"scene_prior_{i}"
            )
            
            scene_data["next_state"] = st.text_input(
                "Next State",
                value=scene.get("next_state", ""),
                key=f"scene_next_{i}"
            )
            
            data["scenes"].append(scene_data)
    
    # Add new scene button
    if st.checkbox("Add New Scene", key="add_new_scene"):
        scene_data = {}
        
        scene_data["scene_id"] = st.text_input(
            "Scene ID",
            value=f"scene_{len(scenes)+1}",
            key="new_scene_id"
        )
        
        scene_data["description"] = st.text_area(
            "Description",
            value="",
            key="new_scene_desc"
        )
        
        scene_data["prior_state"] = st.text_input(
            "Prior State",
            value="",
            key="new_scene_prior"
        )
        
        scene_data["next_state"] = st.text_input(
            "Next State",
            value="",
            key="new_scene_next"
        )
        
        data["scenes"].append(scene_data)
    
    return data


def setting_bible_form(initial_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Form for setting bible entries.
    
    Args:
        initial_data: Initial form data
        
    Returns:
        Dictionary with form data
    """
    data = {}
    
    # Basic info
    data["name"] = st.text_input(
        "Name",
        value=initial_data.get("name", ""),
        help="Name of the setting"
    )
    
    data["description"] = st.text_area(
        "Description",
        value=initial_data.get("description", ""),
        help="General description of the setting"
    )
    
    # Details
    col1, col2 = st.columns(2)
    
    with col1:
        data["time_period"] = st.text_input(
            "Time Period",
            value=initial_data.get("time_period", ""),
            help="When the setting takes place"
        )
    
    with col2:
        data["mood"] = st.text_input(
            "Mood/Atmosphere",
            value=initial_data.get("mood", ""),
            help="Overall feeling of the setting"
        )
    
    # Locations
    st.subheader("Locations")
    
    # Get existing locations
    locations = initial_data.get("locations", [])
    
    # Create container for locations
    data["locations"] = []
    
    # Display existing locations
    for i, location in enumerate(locations):
        cols = st.columns([3, 6, 1])
        
        location_name = location.get("name", "") if isinstance(location, dict) else location
        location_desc = location.get("description", "") if isinstance(location, dict) else ""
        
        with cols[0]:
            new_name = st.text_input(
                "Name",
                value=location_name,
                key=f"loc_name_{i}"
            )
        
        with cols[1]:
            new_desc = st.text_input(
                "Description",
                value=location_desc,
                key=f"loc_desc_{i}"
            )
        
        # Add to data if name is filled
        if new_name:
            data["locations"].append({
                "name": new_name,
                "description": new_desc
            })
    
    # Add new location
    st.caption("Add new location:")
    cols = st.columns([3, 6, 1])
    
    with cols[0]:
        new_name = st.text_input(
            "Name",
            value="",
            key="loc_name_new"
        )
    
    with cols[1]:
        new_desc = st.text_input(
            "Description",
            value="",
            key="loc_desc_new"
        )
    
    # Add to data if name is filled
    if new_name:
        data["locations"].append({
            "name": new_name,
            "description": new_desc
        })
    
    return data


def theme_bible_form(initial_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Form for theme bible entries.
    
    Args:
        initial_data: Initial form data
        
    Returns:
        Dictionary with form data
    """
    data = {}
    
    # Basic info
    data["name"] = st.text_input(
        "Name",
        value=initial_data.get("name", ""),
        help="Theme name or concept"
    )
    
    data["description"] = st.text_area(
        "Description",
        value=initial_data.get("description", ""),
        help="Detailed description of the theme and its significance"
    )
    
    # Symbols
    symbols_input = st.text_input(
        "Symbols",
        value=", ".join(initial_data.get("symbols", [])) if isinstance(initial_data.get("symbols", []), list) else initial_data.get("symbols", ""),
        help="Comma-separated list of symbols representing this theme"
    )
    
    # Parse symbols to list
    if symbols_input:
        data["symbols"] = [symbol.strip() for symbol in symbols_input.split(",") if symbol.strip()]
    else:
        data["symbols"] = []
    
    # Character arcs
    st.subheader("Character Arcs")
    
    # Get existing arcs
    arcs = initial_data.get("character_arcs", {})
    
    # Create container for arcs
    data["character_arcs"] = {}
    
    # Display existing arcs
    for i, (character, arc) in enumerate(arcs.items()):
        cols = st.columns([3, 6, 1])
        
        with cols[0]:
            new_character = st.text_input(
                "Character",
                value=character,
                key=f"arc_char_{i}"
            )
        
        with cols[1]:
            new_arc = st.text_input(
                "Arc",
                value=arc,
                key=f"arc_desc_{i}"
            )
        
        # Only add if both fields are filled
        if new_character and new_arc:
            data["character_arcs"][new_character] = new_arc
    
    # Add new arc
    st.caption("Add new character arc:")
    cols = st.columns([3, 6, 1])
    
    with cols[0]:
        new_character = st.text_input(
            "Character",
            value="",
            key="arc_char_new"
        )
    
    with cols[1]:
        new_arc = st.text_input(
            "Arc",
            value="",
            key="arc_desc_new"
        )
    
    # Add to data if both fields are filled
    if new_character and new_arc:
        data["character_arcs"][new_character] = new_arc
    
    return data


def generic_bible_form(initial_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generic form for bible entries of unknown type.
    
    Args:
        initial_data: Initial form data
        
    Returns:
        Dictionary with form data
    """
    data = {}
    
    # Basic info
    for key, value in initial_data.items():
        if isinstance(value, dict):
            # Skip complex nested objects
            continue
        elif isinstance(value, list):
            # Handle simple lists
            list_input = st.text_input(
                key.capitalize(),
                value=", ".join(str(item) for item in value),
                help=f"Comma-separated {key}"
            )
            
            if list_input:
                data[key] = [item.strip() for item in list_input.split(",") if item.strip()]
            else:
                data[key] = []
        else:
            # Handle simple values
            if isinstance(value, str) and len(value) > 100:
                # Use text area for long text
                data[key] = st.text_area(
                    key.capitalize(),
                    value=value
                )
            else:
                # Use text input for short text
                data[key] = st.text_input(
                    key.capitalize(),
                    value=value
                )
    
    # Allow adding new fields
    st.subheader("Add New Field")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_key = st.text_input("Field Name", key="new_field_name")
    
    with col2:
        new_value = st.text_input("Value", key="new_field_value")
    
    if new_key and new_value:
        data[new_key] = new_value
    
    return data


def track_bible_changes(
    bible_id: int,
    bible_type: str,
    old_data: Dict[str, Any],
    new_data: Dict[str, Any]
):
    """
    Track changes to a bible entry for potential propagation.
    
    Args:
        bible_id: Bible entry ID
        bible_type: Type of bible
        old_data: Previous data
        new_data: New data
    """
    # Character name change (special case that needs propagation)
    if bible_type == "character" and old_data.get("name") != new_data.get("name"):
        add_bible_change(
            bible_id=bible_id,
            bible_type=bible_type,
            field_path="content.name",
            old_value=old_data.get("name"),
            new_value=new_data.get("name")
        )
    
    # Plot title change (special case that needs propagation)
    if bible_type == "plot" and old_data.get("title") != new_data.get("title"):
        add_bible_change(
            bible_id=bible_id,
            bible_type=bible_type,
            field_path="content.title",
            old_value=old_data.get("title"),
            new_value=new_data.get("title")
        )
    
    # Setting name change (special case that needs propagation)
    if bible_type == "setting" and old_data.get("name") != new_data.get("name"):
        add_bible_change(
            bible_id=bible_id,
            bible_type=bible_type,
            field_path="content.name",
            old_value=old_data.get("name"),
            new_value=new_data.get("name")
        ) 