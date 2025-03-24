"""Core data models for the Dramatron system."""
from dataclasses import dataclass, field   
import re
from typing import Dict, List, Optional, Tuple

def extract_elements(text: str, start_marker: str, end_marker: str) -> List[str]:
  """Extracts elements from the text between the start and end markers."""
  elements = []
  start = text.find(start_marker)
  if start == -1:
    raise ValueError("Start marker not found in text")
  start += len(start_marker)
  finish = text.find(end_marker, start)
  if finish == -1:
    raise ValueError("End marker not found after start marker")
  elements.append(text[start:finish].strip())
  return elements

def parse_characters_and_descriptions(text: str) -> List[Dict[str, str]]:
  """Parses characters and descriptions from the text using regex.
  
  Args:
      text: The text containing character and description information
      
  Returns:
      A list of dictionaries mapping character names to their descriptions
  """
  character_blocks = re.split(r'\n\n', text.strip())
  characters_list = []

  for block in character_blocks:
      character_match = re.search(r'\*\*Character:\*\*\s+(.+?)\s+\*\*Description:\*\*', block)
      description_match = re.search(r'\*\*Description:\*\*\s+(.+)', block)
      
      if character_match and description_match:
          character = character_match.group(1).strip()
          description = description_match.group(1).strip()
          characters_list.append({character: description})
          
  return characters_list
      
def parse_places_plot_beats(text: str) -> Tuple[List[str], List[str], List[str]]:
    """Parse places, plot elements, and beats from the text."""
    # Initialize lists to store places, plot elements, and beats
    places_list = []
    plots_list = []
    beats_list = []

    # Regex patterns to match places, plot elements, and beats
    place_pattern = re.compile(r'\*\*Place:\s+(.+?)[\.\*]*\*\*')
    plot_pattern = re.compile(r'[\*\-]\s*Plot element:\s+(.+?)[\.\*]*\*')
    beat_pattern = re.compile(r'[\*\-]\s*Beat:\s*(.+)')

    blocks = text.strip().split('\n\n')

    # Iterate through each block of the text
    for block in blocks:
        place_match = place_pattern.search(block)
        plot_match = plot_pattern.search(block)
        beat_match = beat_pattern.search(block)
        
        if place_match:
            places_list.append(place_match.group(1).strip())
        if plot_match:
            plots_list.append(plot_match.group(1).strip())
        if beat_match:
            beats_list.append(beat_match.group(1).strip())

    # Print the resulting lists
    print("Places List:", places_list)
    print("Plots List:", plots_list)
    
    updated_beats_list = []
    for elem in beats_list:
      elem = elem.replace('* ', '')
      elem = elem.replace('*', '')
      updated_beats_list.append(elem)
    print(f"Beats list: {updated_beats_list}")
    return places_list, plots_list, updated_beats_list


@dataclass
class Title:
  """Title class."""

  title: str

  @classmethod
  def from_string(cls, text: str, title_element: str, end_marker: str):
    # Extract title directly without creating a list for a single element
    title = extract_elements(text, title_element, end_marker)[0]
    return cls(title)

  def to_string(self, title_element: str, end_marker: str):
    s = ''
    s += title_element + self.title
    s += end_marker
    return s
  
@dataclass
class Character():
  """Character class."""

  # Name of the character.
  name: str

  # A single sentence describing the character.
  description: str

  @classmethod
  def from_string(cls, text: str, description_marker: str):
    elements = text.split(description_marker)
    if len(elements) == 2:
      name = elements[0].strip()
      description = elements[1].strip()
      return cls(name, description)
    else:
      return None
    
@dataclass
class Characters():
  """Characters class, containing main characters and their descriptions."""

  # A dictionary of character descriptions.
  character_descriptions: Dict[str, str] = field(default_factory=dict)

  @classmethod
  def from_string(cls, text: str, character_marker: str, stop_marker: str, description_marker: str):
      """Parses the characters from the generated text."""
      text = text.strip()
      # Extracts the character descriptions.
      character_descriptions = {}
      elements = extract_elements(text, character_marker, stop_marker)
      for text_character in elements:
          character = Character.from_string(text_character, description_marker)
          print(character)
          if character is not None:
              character_descriptions[character.name] = character.description

      if character_descriptions == {}:
          elements_custom = parse_characters_and_descriptions(text)
          for element in elements_custom:
              key = list(element.keys())[0]
              value = list(element.values())[0]
              character_descriptions[key] = value
      
      return cls(character_descriptions)

  def to_string(self, character_marker: str, description_marker: str, stop_marker: str, end_marker: str):
    s = '\n'
    for name, description in self.character_descriptions.items():
      s += '\n' + character_marker + ' ' + name + ' ' + description_marker + ' '
      s += description + ' ' + stop_marker + '\n'
    s += end_marker

    return s    

@dataclass
class Place():
  """Place class."""

  # Place name.
  name: str

  # Place description.
  description: str

  @classmethod
  def format_name(cls, name: str):
    if name.find('.') == -1:
      name = name + '.'
    return name

  @classmethod
  def from_string(cls, place_name: str, place_text: str, description_marker: str, end_marker: str):
    place_text += end_marker
    description = extract_elements(place_text, description_marker, end_marker)
    return cls(place_name, description[0])

  @classmethod
  def format_prefix(cls, name, place_marker: str, description_marker: str):
    s = place_marker + name + '\n' + description_marker
    return s

  def to_string(self, place_marker: str, description_marker: str):
    s = self.format_prefix(self.name, place_marker, description_marker) + self.description + '\n\n'
    return s
  

@dataclass
class Scene():
  """Scene class."""

  # The name of the place where the scene unfolds.
  place: str

  # Name of the plot element (e.g., Beginning, Middle, Conclusion).
  plot_element: str

  # A short description of action/story/dramatic event occuring in the scene.
  beat: str

  def to_string(self, place_element: str, plot_element: str, beat_element: str):
    s = place_element + ' ' + self.place + '\n'
    s += plot_element + ' ' + self.plot_element + '\n'
    s += beat_element + ' ' + self.beat + '\n'
    return s
  
@dataclass
class Scenes():
  """Scenes class."""

  # A list of scenes, with place, characters, plot element and beat.
  scenes: List[Scene]

  @classmethod
  def from_string(cls, text: str, place_element: str, plot_element: str, beat_element: str):
    """Parse scenes from generated scenes_text."""

    places = extract_elements(text, place_element, plot_element)
    plot_elements = extract_elements(text, plot_element, beat_element)
    beats = extract_elements(text, beat_element, '\n')

    # Get the number of complete scenes.
    num_complete_scenes = min([len(places), len(plot_elements), len(beats)])

    if num_complete_scenes == 0:
        places, plot_elements, beats = parse_places_plot_beats(text)
    num_complete_scenes = min([len(places), len(plot_elements), len(beats)])
    scenes = []
    for i in range(num_complete_scenes):
      scenes.append(
          Scene(Place.format_name(places[i]), plot_elements[i], beats[i]))
    scenes = cls(scenes)
    return scenes

  def to_string(self, place_element: str, plot_element: str, beat_element: str, end_marker: str):
    s = ''
    for scene in self.scenes:
      s += '\n' + scene.to_string(place_element, plot_element, beat_element)
    s += end_marker
    return s

  def num_places(self):
    return len(set([scene.place for scene in self.scenes]))

  def num_scenes(self) -> int:
    return len(self.scenes)
  

@dataclass
class Story():
  """Story class."""

  # A storyline is a single sentence summary of the whole plot.
  storyline: str

  # A title for the story.
  title: str

  # Map from character names to full descriptions.
  character_descriptions: Dict[str, str]

  # Map from place names to full descriptions.
  place_descriptions: Dict[str, Place]

  # List of scenes.
  scenes: Scenes

  # List of dialogs, one for each scene.
  dialogs: List[str]