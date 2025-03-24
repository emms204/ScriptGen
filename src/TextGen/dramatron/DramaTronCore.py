import datetime
import difflib
import json
import re
import requests
import sys
import time
from typing import Dict, List, NamedTuple, Optional, Union

import collections
import argparse
from HyperParameters import *
from src.TextGen.utils import add_custom_chars_to_prompt


###REPLACE IT WITH SOMETHING OKAYISH
# parser = argparse.ArgumentParser(description ='Process some integers.')
# parser.add_argument('--logline_file', metavar ='loglinefile', 
#                     type = str, nargs ='+',
#                     help ='file to specify the logline')

# args = parser.parse_args()
# print(args)



print('Dramatron hyperparameters set.')



# ------------------------------------------------------------------------------
# Dramatron script entities
# ------------------------------------------------------------------------------


def get_title(title: Title) -> str:
  return title.title



def get_character_description(character: Character) -> str:
  return character.description



def get_character_descriptions(characters: Characters) -> Dict[str, str]:
  return characters.character_descriptions
  

def parse_narrative_elements(output):
    places = []
    plot_elements = []
    beats = []
    
    sections = output.strip().split("\n\n")
    
    for section in sections:
        lines = section.split('\n')
        for line in lines:
            if line.startswith("Place:"):
                place = line.split("Place:")[1].strip()
                places.append(place)
            elif line.startswith("Plot element:"):
                plot_element = line.split("Plot element:")[1].strip()
                plot_elements.append(plot_element)
            elif line.startswith("Beat:"):
                beat = line.split("Beat:")[1].strip()
                beats.append(beat)
    
    return places, plot_elements, beats
  


def get_place_description(place: Place):
  return place.description


# ------------------------------------------------------------------------------
# Rendering of generated stories
def extract_elements(text: str, begin: str, end: str) -> List[str]:
  """Extracts elements from a text string given string and ending markers."""

  results = []
  start = 0
  while True:
    start = text.find(begin, start)
    if start == -1:
      return results
    finish = text.find(end, start)
    if finish == -1:
      return results
    results.append(text[start + len(begin):finish].strip())
    start = finish + len(end)
# ------------------------------------------------------------------------------


def render_story(story: Story) -> str:
  """Render the story in fountain format."""

  lines = []
  lines.append(f'Title: {story.title}')
  lines.append('Author: Co-written by ________ and Dramatron')
  lines.append(
      'Dramatron was developed by Piotr Mirowski and Kory W. Mathewson, '
      'with additional contributions by Juliette Love and Jaylen Pittman, '
      'and is based on a prototype by Richard Evans.')
  lines.append('Dramatron relies on user-provided language models.')
  lines.append('')
  lines.append('====')
  lines.append('')

  lines.append(f'The script is based on the storyline:\n{story.storyline}')
  lines.append('')
  if story.character_descriptions is not None:
    for name, description in story.character_descriptions.items():
      lines.append(f'{name}: {description}')
      lines.append('')

  # For each scene, render scene information.
  if story.scenes is not None:
    scenes = story.scenes.scenes
    for i, scene in enumerate(scenes):
      lines.append(f'Scene {i+1}')
      lines.append(f'{PLACE_ELEMENT}{scene.place}')
      lines.append(f'{PLOT_ELEMENT}{scene.plot_element}')
      lines.append(f'{BEAT_ELEMENT}{scene.beat}')
      lines.append('')
  else:
    scenes = []

  lines.append('====')
  lines.append('')

  # For each scene, render the scene's place description, characters and dialog.
  for i, scene in enumerate(scenes):

    # Output the places and place descriptions.
    lines.append(f'INT/EXT. {scene.place} - Scene {i+1}')
    place_descriptions = story.place_descriptions
    if (not place_appears_earlier(scene.place, story, i) and
        place_descriptions is not None and scene.place in place_descriptions):
      lines.append('')
      lines.append(get_place_description(place_descriptions[scene.place]))

    # Output the characters and descriptions.
    lines.append('')
    for c in story.character_descriptions.keys():
      if c in scene.beat and not character_appears_earlier(c, story, i):
        lines.append(story.character_descriptions[c])

    # Output the dialog.
    if story.dialogs is not None and len(story.dialogs) > i:
      lines.append('')
      lines_dialog = strip_remove_end(str(story.dialogs[i]))
      lines.append(lines_dialog)
      lines.append('')
      lines.append('')

  return '\n'.join(lines)


def place_appears_earlier(place: str, story: Story, index: int) -> bool:
  """Return True if the place appears earlier in the story."""

  for i in range(index):
    scene = story.scenes.scenes[i]
    if scene.place == place:
      return True
  return False


def character_appears_earlier(character: str, story: Story, index: int) -> bool:
  """Return True if the character appears earlier in the story."""

  for i in range(index):
    scene = story.scenes.scenes[i]
    if character in scene.beat:
      return True
  return False


def render_prompts(prompts):
  """Render the prompts."""

  def _format_prompt(prompt, name):
    prompt_str = '=' * 80 + '\n'
    prompt_str += 'PROMPT (' + name + ')\n'
    prompt_str += '=' * 80 + '\n\n'
    prompt_str += str(prompt) + '\n\n'
    return prompt_str

  prompts_str = _format_prompt(prompts['title'], 'title')
  prompts_str += _format_prompt(prompts['characters'], 'characters')
  prompts_str += _format_prompt(prompts['scenes'], 'scenes')
  places = prompts['places']
  if places is not None:
    for k, prompt in enumerate(places):
      prompts_str += _format_prompt(prompt, 'place ' + str(k + 1))
  dialogs = prompts['dialogs']
  if dialogs is not None:
    for k, prompt in enumerate(dialogs):
      prompts_str += _format_prompt(prompt, 'dialog ' + str(k + 1))
  return prompts_str


# ------------------------------------------------------------------------------
# Language API definition
# ------------------------------------------------------------------------------




# ------------------------------------------------------------------------------
# Dramatron Generator
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------------


class GenerationAction:
  NEW = 1
  CONTINUE = 2
  REWRITE = 3


class GenerationHistory:
  """Custom data structure to handle the history of GenerationAction edits:

  NEW, CONTINUE or REWRITE. Consecutive REWRITE edits do not add to history.
  """

  def __init__(self):
    self._items = []
    self._actions = []
    self._idx = -1
    self._locked = False

  def _plain_add(self, item, action: GenerationAction):
    self._items.append(item)
    self._actions.append(action)
    self._idx = len(self._items) - 1
    return self._idx

  def add(self, item, action: GenerationAction):
    if len(self._items) == 0 or action != GenerationAction.REWRITE:
      return self._plain_add(item, action)
    last_action = self._actions[-1]
    if last_action != GenerationAction.REWRITE:
      return self._plain_add(item, action)
    self._items[self._idx] = item
    return self._idx

  def previous(self):
    if len(self._items) == 0:
      return None
    self._idx = max(self._idx - 1, 0)
    return self._items[self._idx]

  def next(self):
    if len(self._items) == 0:
      return None
    self._idx = min(self._idx + 1, len(self._items) - 1)
    return self._items[self._idx]

filter = None

print('Dramatron set-up complete.')





print(f'New Dramatron generator created.')

story = None

#@ Title gen   ????????????????????????????????

# data_title = {"text": "", "text_area": None, "seed": generator.seed - 1}

def fun_generate_title(data_title, generator):
  data_title["seed"] += 1
  seed = data_title["seed"]
  generator.step(0, seed=seed)
  data_title["text"] = generator.title_str().strip()
  # return data_title['text'], data_title

def fun_rewrite_title(text, generator):
  text_to_parse = TITLE_ELEMENT + text + END_MARKER
  generator.rewrite(text_to_parse, level=1)
  return text

# Widget to generate new title. _________________________________________________ GENERATING NEW TITLE
#fun_generate_title

# Widget to rewrite the title. ______________________________________________________REWRITE THE TITLE

#fun_rewrite_title
# fun_generate_title(new_title_button)


######################################################################################################################################################################################################################################################################

#@title Generate **Characters**

# data_chars = {"text": "", "text_area": None, "seed": generator.seed - 1,
#               "history": GenerationHistory(), "lock": False}

def fun_generate_characters(data_chars, generator, predefined_chars = False, initial_char_data = ''):
  data_chars["seed"] += 1
  seed = data_chars["seed"]
  data_chars["lock"] = True
  while True:
    generator.step(1, seed=seed, predefined_chars = predefined_chars, initial_char_data = initial_char_data)
    data_chars["text"] = strip_remove_end(generator.characters.to_string())
    if len(data_chars["text"]) == 0:
      seed += 1
    else:
      break
  data_chars["seed"] = seed
  data_chars["history"].add(data_chars["text"], GenerationAction.NEW)
  data_chars["lock"] = False
  # return data_chars['text'], data_chars

def fun_continue_characters(data_chars, generator):
  data_chars["seed"] += 1
  seed = data_chars["seed"]
  data_chars["lock"] = True
  generator.complete(level=2, seed=seed, sample_length=256)
  data_chars["text"] = strip_remove_end(generator.characters.to_string())
  data_chars["history"].add(data_chars["text"], GenerationAction.CONTINUE)
  # if data_chars["text_area"] is not None:
  #   data_chars["text_area"].value = data_chars["text"]
  data_chars["lock"] = False
  # return data_chars['text'], data_chars

def fun_back_forward(data, history: GenerationHistory, delta: int, data_places):
  data["lock"] = True
  if delta > 0:
    data["text"] = history.next()
  if delta < 0:
    data["text"] = history.previous()
  # if data["text"] is not None and data["text_area"] is not None:
  #     data["text_area"].value = data["text"]
  data["lock"] = False
  # return data['text'], data_places

def fun_back_characters(data_chars):
  fun_back_forward(data_chars, data_chars["history"], -1)

def fun_forward_characters(data_chars):
  fun_back_forward(data_chars, data_chars["history"], 1)

def fun_rewrite_characters(text, data_chars, generator):
  data_chars["text"] = text
  text_to_parse = text + END_MARKER
  generator.rewrite(text_to_parse, level=2)
  if data_chars["lock"] is False:
    data_chars["history"].add(text, GenerationAction.REWRITE)
  return text

# Widget to generate new characters.
#fun_generate_characters

# Widget to continue the generation of the current characters.
#fun_continue_characters

# Widgets to move back and forward in history of generation.
#fun_back_characters
#fun_forward_characters


# Render the characters using widgets.
#REWRITE THAT 
# data_chars["text_area"] = widgets.Textarea(
#     value=data_chars["text"], layout=layout, description=' ',
#     style={'description_width': 'initial'})
# textarea_chars = widgets.interactive(
#     fun_rewrite_characters, text=data_chars["text_area"])
# display(textarea_chars)

# # Trigger generation for first seed.
# fun_generate_characters(new_characters_button)




### ???????????????????
#@title Generate a **Plot Synopsis** (sequence of **Scenes**)

# data_scenes = {"text": None, "text_area": None, "seed": generator.seed - 1,
#                "history": GenerationHistory(), "lock": False}

def fun_generate_scenes(data_scenes, generator):
  data_scenes["seed"] += 1
  seed = data_scenes["seed"]
  data_scenes["lock"] = True
  generator.step(2, seed=seed)
  data_scenes["text"] = strip_remove_end(generator.scenes.to_string())
  data_scenes["history"].add(data_scenes["text"], GenerationAction.NEW)

  if data_scenes["text_area"] is not None:
    data_scenes["text_area"].value = data_scenes["text"]
  data_scenes["lock"] = False
  # return data_scenes['text'], data_scenes
  

def fun_continue_scenes(data_scenes, generator):
  data_scenes["seed"] += 1
  seed = data_scenes["seed"]
  data_scenes["lock"] = True

  generator.complete(level=3, seed=seed, sample_length=256)
  data_scenes["text"] = strip_remove_end(generator.scenes.to_string())
  data_scenes["history"].add(data_scenes["text"], GenerationAction.CONTINUE)

  if data_scenes["text_area"] is not None:
    data_scenes["text_area"].value = data_scenes["text"]
  data_scenes["lock"] = False
  # return data_scenes['text'], data_scenes

def fun_back_scenes(data_scenes):
  fun_back_forward(data_scenes, data_scenes["history"], -1)

def fun_forward_scenes(data_scenes):
  fun_back_forward(data_scenes, data_scenes["history"], 1)

def fun_rewrite_scenes(text, generator, data_scenes):
  generator.rewrite(text, level=3)
  if data_scenes["lock"] is False:
    data_scenes["history"].add(text, GenerationAction.REWRITE)
  return text

# Widget to generate new scenes.
#fun_generate_scenes

# Widget to continue the generation of the current scenes.
#fun_continue_scenes

# Widgets to move back and forward in history of generation.
#fun_back_scenes
#fun_forward_scenes


# # Render the scenes using widgets.
# layout = widgets.Layout(height='590px', min_height='600px', width='auto')
# data_scenes["text_area"] = widgets.Textarea(
#     value=data_scenes["text"], layout=layout, description=' ',
#     style={'description_width': 'initial'})
# scanes_textarea = widgets.interactive(
#     fun_rewrite_scenes, text=data_scenes["text_area"])
# display(scanes_textarea)

# Trigger generation for first seed.
# fun_generate_scenes(new_scenes_button)

##################################################################################################################################################################################################################################################
######???????????????????????????????????
#@title Generate **Place Descriptions**

#@markdown This cell generates a description for each **Place** name in the **Plot Synopsis**. If you edit place names in the **Plot Synopsis** _after_ having run this cell, you will need to re-run this cell to update **Places**.
###################################################################################################################################################################################################################################################


# place_names = list(set([scene.place for scene in generator.scenes[0]]))
# place_descriptions = {place_name: Place(place_name, '')
#                       for place_name in place_names}
# data_places = {"descriptions": place_descriptions, "text_area": {},
#                "seed": generator.seed - 1}

def fun_generate_places(data_places, generator):
  data_places["seed"] += 1
  seed = data_places["seed"]

  # Generate all the places.
  generator.step(3, seed=seed)
  print(generator.places)
  data_places["descriptions"] = generator.places
  missing_places = {k: True for k in data_places["text_area"].keys()}
  for place_name, place_description in data_places["descriptions"].items():
    if place_name in data_places["text_area"]:
      description = place_description.description
      data_places["text_area"][place_name].value = description
      del missing_places[place_name]
    else:
        pass
      # print(f"\nWarning: [{place_name}] was added to the plot synopsis.")
      # print(f"Make a copy of the outputs and re-run the cell.")
  for place_name in missing_places:
    data_places["text_area"][place_name].value = (
        f"Warning: [{place_name}] was removed from the plot synopsis. "
        "Make a copy of the outputs and re-run the cell.")
    
  # return data_places

def fun_rewrite_places(place_name, text, generator):
  generator.rewrite(text, level=4, entity=place_name)
  return text

# Widget to generate new scenes.
#fun_generate_places


# Trigger generation for first seed.
#fun_generate_places(new_places_button)

##################################################################################################################################################################################################
####??????????????????????????
#@title Generate **Dialogues**
################################################################################################################################################################################################

# num_scenes =  generator.num_scenes()

# data_dialogs = {
#     "lock": False,
#     "text_area": None,
#     "seed": generator.seed - 1,
#     "history": [GenerationHistory() for _ in range(99)],
#     "scene": 1
# }

# Ensure idx_dialog is initialize
# idx_dialog = data_dialogs["scene"] - 1

def fun_generate_dialog(data_dialogs, generator):
  data_dialogs["seed"] += 1
  seed = data_dialogs["seed"]
  idx_dialog = data_dialogs["scene"] - 1
  data_dialogs["lock"] = True

  generator.step(4, seed=seed, idx=idx_dialog)
  data_dialogs["history"][idx_dialog].add(
      generator.dialogs[idx_dialog], GenerationAction.NEW)

  if data_dialogs["text_area"] is not None:
    data_dialogs["text_area"].value = generator.dialogs[idx_dialog]
  data_dialogs['raw_text'] = generator.dialogs[idx_dialog]
  
  data_dialogs["lock"] = False

  return data_dialogs

def fun_load_dialog(scene, generator, data_dialogs):
  idx_dialog = scene - 1
  scene_exists = (
      len(generator.dialogs) > idx_dialog and
      len(generator.dialogs[idx_dialog]) > 0)
  # Update existing text area with a waiting message or load existing scene.
  if scene_exists:
    data_dialogs["lock"] = True
    if data_dialogs["text_area"] is not None:
      data_dialogs["text_area"].value = generator.dialogs[idx_dialog]
    data_dialogs["scene"] = scene
    data_dialogs["lock"] = False
  else:
    data_dialogs["scene"] = scene
    fun_generate_dialog(None)

def fun_continue_dialog(data_dialogs, generator):
  data_dialogs["seed"] += 1
  seed = data_dialogs["seed"]
  idx_dialog = data_dialogs["scene"] - 1
  data_dialogs["lock"] = True
  generator.complete(level=5, seed=seed, entity=idx_dialog,
                     sample_length=SAMPLE_LENGTH)
  data_dialogs["history"][idx_dialog].add(
      generator.dialogs[idx_dialog], GenerationAction.CONTINUE)
  if data_dialogs["text_area"] is not None:
    data_dialogs["text_area"].value = generator.dialogs[idx_dialog]
  data_dialogs["lock"] = False

def fun_back_dialog(data_dialogs):
  idx_dialog = data_dialogs["scene"] - 1
  if idx_dialog >= 0 and idx_dialog < len(data_dialogs["history"]):
    fun_back_forward(data_dialogs, data_dialogs["history"][idx_dialog], -1)

def fun_forward_dialog(data_dialogs):
  idx_dialog = data_dialogs["scene"] - 1
  if idx_dialog >= 0 and idx_dialog < len(data_dialogs["history"]):
    fun_back_forward(data_dialogs, data_dialogs["history"][idx_dialog], 1)

# Function to edit the specific dialog.
def fun_rewrite_dialog(text, data_dialogs, generator):
  if data_dialogs["lock"] == False:
    idx_dialog = data_dialogs["scene"] - 1
    generator.rewrite(text, level=5, entity=idx_dialog)
  return text

# Widget to choose a seed and generate new scenes.
#fun_load_dialog, scene=scene_slider

# Widget to generate new dialogue.
# fun_generate_dialog

# Widget to continue the generation of the current dialogue.
#fun_continue_dialog

# Widgets to move back and forward in history of generation.

#fun_back_dialog
#fun_forward_dialog

# Organise the widgets.

# Render the dialog using widgets.
#fun_rewrite_dialog

# Trigger generation for first seed.
#fun_generate_dialog()


################################################################################################################################################
########????????????????
#@title Render the script
###############################################################################################################################################
#@markdown Run this cell to render the whole story as a text string. This cells also renders all the edits and prefixes as text strings.

# # Render the story.
# story = generator.get_story()
# script_text = render_story(story)
# print(script_text)

# # Render the prompts.
# prefix_text = render_prompts(generator.prompts)

# # Render the interventions.
# edits_text = ''
# for timestamp in sorted(generator.interventions):
#   edits_text += 'EDIT @ ' + str(timestamp) + '\n'
#   edits_text += generator.interventions[timestamp] + '\n\n\n'

# # Prepare the filenames for saving the story and prompts.
# timestamp_generation = datetime.datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')
# title_ascii = re.sub('[^0-9a-zA-Z]+', '_',
#                      generator.title_str().strip()).lower()
# filename_script = f'{title_ascii}_{timestamp_generation}_script.txt'
# filename_prefix = f'{title_ascii}_{timestamp_generation}_prefix.txt'
# filename_edits = f'{title_ascii}_{timestamp_generation}_edits.txt'
# filename_config = f'{title_ascii}_{timestamp_generation}_config.json'





