from typing import Dict, List, Optional, Tuple
from .models import Title, Character, Characters, Scene, Scenes, Place, Story
from .language_api import LanguageAPI, FilterAPI
import time
import collections
import difflib
from src.TextGen.utils import add_custom_chars_to_prompt
from src.TextGen.HyperParameters import DIALOG_MARKER, PLACE_ELEMENT, DESCRIPTION_ELEMENT, CHARACTERS_ELEMENT, PLOT_ELEMENT, BEAT_ELEMENT, SUMMARY_ELEMENT, PREVIOUS_ELEMENT, END_MARKER

def detect_loop(text: str, max_num_repetitions: int):
  """Detect loops in generated text."""

  blocks = text.split('\n\n')
  num_unique_blocks = collections.Counter(blocks)
  for block in blocks:
    num_repetitions = num_unique_blocks[block]
    if num_repetitions > max_num_repetitions:
      print(f'Detected {num_repetitions} repetitions of block:\n{block}')
      return True
  return False

def prefix_summary(storyline: str,
                   scenes: List[Scene],
                   summary_element: str,
                   previous_element: str) -> str:
    """Assemble the summary part of the dialog prefix."""

    summary = summary_element + storyline + '\n'
    if len(scenes) > 1:
        summary += previous_element + scenes[len(scenes) - 2].beat + '\n'
    return summary

def strip_remove_end(text: str, end_marker: str) -> str:
    text = text.strip()

    end_marker_stripped = end_marker.strip()
    if text.endswith(end_marker_stripped):
        text = text[:-len(end_marker_stripped)]
    return text


def diff_prompt_change_str(prompt_before: str, prompt_after: str) -> str:
  """Return a text diff on prompt sets `prompt_before` and `prompt_after`."""

  # For the current element, compare prompts line by line.
  res = difflib.unified_diff(
      prompt_before.split('\n'), prompt_after.split('\n'))
  diff = ''
  for line in res:
    line = line.strip()
    if line != '---' and line != '+++' and not line.startswith('@@'):
      if len(line) > 1 and (line.startswith('+') or line.startswith('-')):
        diff += line + '\n'
  if diff.endswith('\n'):
    diff = diff[:-1]
  return diff


def diff_prompt_change_list(prompt_before: List[str],
                            prompt_after: List[str]) -> str:
  """Return a text diff on prompt sets `prompt_before` and `prompt_after`."""

  # Handle deletions and insertions.
  len_before = len(prompt_before)
  len_after = len(prompt_after)
  if len_before > len_after:
    return 'Deleted element'
  if len_before < len_after:
    return 'Added new element'

  diffs = [
      diff_prompt_change_str(a, b)
      for (a, b) in zip(prompt_before, prompt_after)
  ]
  return '\n'.join([diff for diff in diffs if len(diff) > 0])


def diff_prompt_change_scenes(prompt_before: List[Scene],
                              prompt_after: List[Scene]) -> str:
  """Return a text diff on prompt sets `prompt_before` and `prompt_after`."""

  # Handle deletions and insertions.
  len_before = len(prompt_before)
  len_after = len(prompt_after)
  if len_before > len_after:
    return 'Deleted element'
  if len_before < len_after:
    return 'Added new element'

  diffs = [
      diff_prompt_change_list([a.place, a.plot_element, a.beat],
                              [b.place, b.plot_element, b.beat])
      for (a, b) in zip(prompt_before, prompt_after)
  ]
  return '\n'.join([diff for diff in diffs if len(diff) > 0])


def diff_prompt_change_dict(prompt_before: Dict[str, str],
                            prompt_after: Dict[str, str]) -> str:
  """Return a text diff on prompt sets `prompt_before` and `prompt_after`."""

  # Loop over the keys in the prompts to compare them one by one.
  keys_before = sorted(prompt_before.keys())
  keys_after = sorted(prompt_after.keys())
  diffs = [
      diff_prompt_change_str(a, b) for (a, b) in zip(keys_before, keys_after)
  ]
  diff_keys = '\n'.join([diff for diff in diffs if len(diff) > 0])
  # Loop over the values in the prompts to compare them one by one.
  values_before = sorted(prompt_before.values())
  values_after = sorted(prompt_after.values())
  diffs = [
      diff_prompt_change_str(a, b)
      for (a, b) in zip(values_before, values_after)
  ]
  diff_values = '\n'.join([diff for diff in diffs if len(diff) > 0])
  return diff_keys + diff_values



class StoryGenerator:
    """Generates a complete story step by step."""
    
    def __init__(
        self,
        storyline: str,
        prefixes: Dict[str, str],
        client: LanguageAPI,
        filter: Optional[FilterAPI] = None,
        max_paragraph_length: int = 1024,
        num_samples: int = 1,
    ):
        """Initialize the StoryGenerator."""
        self.storyline = storyline
        self.prefixes = prefixes
        self.client = client
        self.filter = filter
        self.max_paragraph_length = max_paragraph_length
        self.num_samples = num_samples
        
        # Story components
        self.title = Title()
        self.characters = Characters()
        self.scenes = Scenes([])
        self.places = {}
        self.dialogs = []
        
        # History tracking
        self.interventions = {}

    def generate_text(self, generation_prompt: str,
                      end_marker: str,
                      max_num_attempts: int,
                      sample_length: Optional[int] = None,
                      max_paragraph_length: int = 1024,
                      seed: Optional[int] = None,
                      max_num_repetitions: Optional[int] = None) -> str:
        """ Generate text using the generation prompt."""

        # Set default sample length if not provided
        if sample_length is None:
           sample_length = self.client.default_sample_length

        # Calculate maximum number of API calls
        max_num_calls = max_paragraph_length // sample_length + 1
        num_calls = 0

        # Initialize variables for the result and seed
        result = ''
        current_seed = seed or 0

        # Begin text generation loop
        while True:
            # Append current result to prompt
            prompt = generation_prompt + result

            # Generate a sample from the language model
            responses = self.client.sample(
                prompt=prompt,
                sample_length=sample_length,
                seed=current_seed,
                num_samples=self.num_samples
            )
            response = responses[0]

            # Check if the response is filtered
            if filter is not None and not filter.validate(response.text):
                return 'Content was filtered out.' + end_marker

            # Check for loops in the generated text
            if max_num_repetitions and detect_loop(
               response.text, max_num_repetitions=max_num_repetitions):
                # Increment seed to generate a different sample
                current_seed += 1
                if current_seed > (seed + max_num_attempts):
                    break
            else:
                # Append response text to result
                result += response.text
                num_calls += 1

                # Break if the result exceeds the maximum paragraph length
                if (max_paragraph_length is not None and
                        len(result) > max_paragraph_length):
                    break

                # Break if the number of API calls exceeds the maximum allowed
                if num_calls >= max_num_calls:
                    break

        # Return the generated text with the end marker
        return result + end_marker


    def generate_text_no_loop(self, generation_prompt: str,
                            sample_length: Optional[int] = None,
                            max_paragraph_length: int = 1024,
                            seed: Optional[int] = None) -> str:
        """Generate text using the generation prompt, without any loop."""
        return self.generate_text(
            generation_prompt=generation_prompt,
            end_marker='**END**',
            max_num_attempts=4,
            sample_length=sample_length,
            max_paragraph_length=sample_length,
            seed=seed,
            max_num_repetitions=None,
            num_samples=self.num_samples)


    def generate_title(self, storyline: str,
                       title_element:str,
                    prefixes: Dict[str, str],
                    seed: Optional[int] = None
                    ):
        """Generate a title given a storyline, and client."""

        # Combine the prompt and storyline as a helpful generation prefix
        titles_prefix = prefixes['TITLES_PROMPT'] + storyline + ' ' + title_element
        title_text = self.generate_text_no_loop(
            generation_prompt=titles_prefix,
            seed=seed)
        title = Title.from_string(title_element + title_text)
        return (title, titles_prefix)


    def generate_characters(
        self,
        storyline: str,
        prefixes: Dict[str, str],
        seed: Optional[int] = None,
        predefined_chars = False, char_data = ''):
        """Generate characters given a storyline, prompt, and client."""

        # Combine the prompt and storyline as a helpful generation prefix
        if predefined_chars == False:
            characters_prefix = prefixes['CHARACTERS_PROMPT'] + storyline
            characters_text = self.generate_text(
                generation_prompt=characters_prefix,
                seed=seed)
            characters = Characters.from_string(characters_text)

        elif predefined_chars == True:
            characters_prefix = add_custom_chars_to_prompt(prefixes, char_data)['CHARACTERS_PROMPT_CUSTOM'] + storyline
            characters_text = self.generate_text(generation_prompt=characters_prefix,
                                            seed = seed)
            characters = Characters.from_string(characters_text)

        return (characters, characters_prefix)


    def generate_scenes(self,
                        storyline: str,
                        scenes_marker: str,
                        character_descriptions: Dict[str, str],
                        prefixes: Dict[str, str],
                        seed: Optional[int] = None,
                        ):
        """Generate scenes given storyline, prompt, main characters, and client."""
        scenes_prefix = prefixes['SCENE_PROMPT'] + storyline + '\n'
        for name in character_descriptions:
            scenes_prefix += character_descriptions[name] + '\n'
        scenes_prefix += '\n' + scenes_marker
        scenes_text = self.generate_text(
            generation_prompt=scenes_prefix,
            seed=seed)
        scenes = Scenes.from_string(scenes_text)

        return (scenes, scenes_prefix)


    def generate_place_descriptions(self,
                                    storyline: str,
                                    scenes: Scenes,
                                    prefixes: Dict[str, str],
                                    seed: Optional[int] = None,
                                    ):
        """Generate a place description given a scene object and a client."""

        place_descriptions = {}

        # Get unique place names from the scenes.
        unique_place_names = set([scene.place for scene in scenes.scenes])

        # Build a unique place prefix prompt.
        place_prefix = prefixes['SETTING_PROMPT'] + storyline + '\n'

        # Build a list of place descriptions for each place
        place_prefixes = []
        for place_name in unique_place_names:
            place_suffix = Place.format_prefix(place_name)
            place_text = self.generate_text(
                generation_prompt=place_prefix + place_suffix,
                seed=seed)
            place_text = place_suffix + place_text
            place_descriptions[place_name] = Place.from_string(place_name, place_text)
            place_prefixes.append(place_prefix + place_suffix)

        return (place_descriptions, place_prefixes)
    
    def generate_dialog(self,
                        storyline: str,
                        scenes: List[Scene],
                        character_descriptions: Dict[str, str],
                        place_descriptions: Dict[str, Place],
                        prefixes: Dict[str, str],
                        dialog_marker: str,
                        place_element: str,
                        description_element: str,
                        characters_element: str,
                        plot_element: str,
                        beat_element: str,
                        summary_element: str,
                        previous_element: str,
                        seed: Optional[int] = None,
                        ):
        """Generate dialog given a scene object and a client."""

        # Use the last scene from the list of scenes
        scene = scenes[-1]

        # Retrieve the place description for the current scene's place, if available
        place_description = place_descriptions.get(scene.place)

        # Construct the place text using the place element, place name, and its description
        place_t = (place_element + scene.place + '\n' +
                   (place_description.description and description_element + place_description.description + '\n'))

        # Construct the characters text by iterating over character descriptions
        # Include a character's description if their name is present in the scene's beat
        characters_t = characters_element + ''.join(
            name in scene.beat and character_descriptions[name] + '\n'
            for name in character_descriptions)

        # Construct the plot element text
        plot_element_t = plot_element + scene.plot_element + '\n'

        # Generate a summary using the prefix_summary helper function
        summary_t = prefix_summary(
            storyline, scenes, summary_element, previous_element)

        # Construct the beat text
        beat_t = beat_element + scene.beat + '\n'

        # Concatenate all elements to form the dialog prefix
        dialog_prefix = (prefixes['DIALOG_PROMPT'] + place_t + characters_t +
                         plot_element_t + summary_t + beat_t)

        # Add the dialog marker to the prefix
        dialog_prefix += '\n' + dialog_marker + '\n'

        # Generate the dialog using the constructed dialog prefix
        dialog = self.generate_text(
            generation_prompt=dialog_prefix,
            seed=seed)

        # Return the generated dialog and the prefix used
        return (dialog, dialog_prefix)
    
    def get_story(self):
        if self.characters is not None:
           character_descriptions = self.characters.character_descriptions
        else:
           character_descriptions = None
        return Story(
            storyline=self.storyline,
            title=self.title.title,
            character_descriptions=character_descriptions,
            place_descriptions=self.places,
            scenes=self.scenes,
            dialogs=self.dialogs)

    
    def step(self,
             level: Optional[int] = None,
             seed: Optional[int] = None,
             idx: Optional[int] = None, 
             predefined_chars=False,
             initial_char_data='') -> bool:
        """Step down a level in the hierarchical generation of a story."""

        # Check and update the current level of story generation.
        if level is None:
            level = self._level
        if level < 0 or level >= len(self.level_names):
            raise ValueError('Invalid level encountered on step.')
        level += 1
        self._level = level

        # Record the timestamp and intervention information for the current step.
        timestamp = time.time()
        self.interventions[timestamp] = 'STEP ' + str(level) + '\n'
        print(f'PREDEFINED CHARS: {predefined_chars}')

        if level == 1:
            # Step 1: Generate the story title based on the storyline.
            (title, titles_prefix) = self.generate_title(
                storyline=self._storyline,
                prefixes=self._prefixes,
                seed=seed)
            self._title = title
            self.prompts['title'] = titles_prefix
            self.interventions[timestamp] += title.to_string()
            # Determine success by checking if the title has content.
            success = len(title.title) > 0
            return success

        if level == 2 and predefined_chars:
            # Step 2: Use predefined characters if specified.
            (characters, character_prompts) = self.generate_characters(
                storyline=self._storyline,
                prefixes=self._prefixes,
                seed=seed,
                predefined_chars=predefined_chars, char_data=initial_char_data)
            self._characters = characters
            self.prompts['characters'] = character_prompts
            self.interventions[timestamp] += characters.to_string()
            # Determine success by checking if characters have descriptions.
            success = len(characters.character_descriptions) > 0
            return success

        elif level == 2:
            # Step 2: Generate characters based on the storyline.
            (characters, character_prompts) = self.generate_characters(
                storyline=self._storyline,
                prefixes=self._prefixes,
                seed=seed)
            self._characters = characters
            self.prompts['characters'] = character_prompts
            self.interventions[timestamp] += characters.to_string()
            # Determine success by checking if characters have descriptions.
            success = len(characters.character_descriptions) > 0
            return success

        if level == 3:
            # Step 3: Generate a sequence of scenes using the storyline and characters.
            scenes, scene_prompts = self.generate_scenes(
                storyline=self._storyline,
                character_descriptions=self._characters.character_descriptions,
                prefixes=self._prefixes,
                seed=seed)
            self._scenes = scenes
            self.prompts['scenes'] = scene_prompts
            self.interventions[timestamp] += scenes.to_string()
            # Determine success by checking if scenes have been generated.
            success = len(scenes.scenes) > 0
            return success

        if level == 4:
            # Step 4: Generate place descriptions for each scene.
            place_descriptions, place_prompts = self.generate_place_descriptions(
                storyline=self._storyline,
                scenes=self._scenes,
                prefixes=self._prefixes,
                seed=seed)
            self._places = place_descriptions
            self.prompts['places'] = place_prompts
            for place_name in place_descriptions:
                place = place_descriptions[place_name]
                if place:
                    self.interventions[timestamp] += place.to_string()
            # Determine success by checking the number of places described matches scenes.
            num_places = self._scenes.num_places()
            success = (len(place_descriptions) == num_places) and num_places > 0
            return success

        if level == 5:
            # Step 5: For each scene, generate dialogue based on scene information.
            dialogs, dialog_prompts = zip(*[
                self.generate_dialog(
                    storyline=self._storyline,
                    scenes=self._scenes.scenes[:(k + 1)],
                    character_descriptions=(self._characters.character_descriptions),
                    place_descriptions=self._places,
                    prefixes=self._prefixes,
                    dialog_marker=DIALOG_MARKER,
                    place_element=PLACE_ELEMENT,
                    description_element=DESCRIPTION_ELEMENT,
                    characters_element=CHARACTERS_ELEMENT,
                    plot_element=PLOT_ELEMENT,
                    beat_element=BEAT_ELEMENT,
                    summary_element=SUMMARY_ELEMENT,
                    previous_element=PREVIOUS_ELEMENT,
                    seed=seed) for k in range(len(self._scenes.scenes))
            ])
            self._dialogs = dialogs
            self.prompts['dialogs'] = dialog_prompts
            for dialog in dialogs:
                self.interventions[timestamp] += str(dialog)
            # Dialog generation is considered successful by default.
            return True

  
    def rewrite(self, text, level=0, entity=None):
        """
        Rewrite the story at a given level with the provided text.

        :param text: The text to rewrite the story with.
        :param level: The level of the story to rewrite. 0 is the storyline, 1 is the title, 2 is the characters, 3 is the scenes, 4 is the places, and 5 is the dialog.
        :param entity: The specific entity to rewrite at the given level. For example, if level is 4 (places), then entity is the name of the place.
        """
        if level < 0 or level >= len(self.level_names):
            raise ValueError('Invalid level encountered on step.')

        if level == 0:
            # Step 0: Rewrite the storyline and begin new story.
            # This just rewrites the storyline and doesn't touch any other part of the story.
            self._set_storyline(text)

        if level == 1:
            # Step 1: Rewrite the title.
            # This just rewrites the title and doesn't touch any other part of the story.
            self._title = Title.from_string(text)

        if level == 2:
            # Step 2: Rewrite the characters.
            # This just rewrites the characters and doesn't touch any other part of the story.
            self._characters = Characters.from_string(text)

        if level == 3:
            # Step 3: Rewrite the sequence of scenes.
            # This just rewrites the scenes and doesn't touch any other part of the story.
            self._scenes = Scenes.from_string(text)

        if level == 4:
            # Step 4: For a given place, rewrite its place description.
            # This just rewrites the place description for the given place and doesn't touch any other part of the story.
            place_descriptions = self._places
            if entity in place_descriptions:
                self._places[entity] = Place.from_string(entity, text)

        if level == 5:
            # Step 5: Rewrite the dialog of a given scene.
            # This just rewrites the dialog for the given scene and doesn't touch any other part of the story.
            dialogs = self._dialogs
            num_scenes = len(self._scenes.scenes)
            if entity >= 0 and entity < num_scenes:
                self._dialogs[entity] = text

            # Keep track of each rewrite intervention.
            timestamp = time.time()
            self.interventions[timestamp] = 'REWRITE ' + self.level_names[level]
            if entity:
                self.interventions[timestamp] += ' ' + str(entity)

    def complete(self,
               level=0,
               seed=None,
               entity=None):
        """Complete the story generation to the given level.

        level: The level of completion. 0 is the storyline, 1 is the title,
            2 is the characters, 3 is the scenes, 4 is the places, and 5 is the
            dialogs.

        seed: The random seed to use for generation. If None, use the default
            seed.

        entity: The entity to generate if level is 4 or 5. For example, if level
            is 4, then entity is the index of the place to generate. If level is 5,
            then entity is the index of the dialog to generate.
        """
        if level < 0 or level >= len(self.level_names):
            raise ValueError('Invalid level encountered on step.')
        if level == 2:
            # Step 2: Complete the characters.
            # Get the text of the current characters.
            text_characters = self._characters.to_string()
            # Strip away any trailing newlines.
            text_characters = strip_remove_end(text_characters)
            # Build the prompt by combining the prompt for characters with the
            # current characters.
            prompt = self.prompts['characters'] + text_characters
            # Generate the new characters.
            text = self.generate_text(
                generation_prompt=prompt,
                seed=seed)
            # Create a new Characters object from the generated text and the
            # current characters.
            new_characters = Characters.from_string(text_characters + text)
            # Compute the diff between the old and new characters.
            prompt_diff = diff_prompt_change_dict(
                self._characters.character_descriptions,
                new_characters.character_descriptions)
            # Set the characters to the new characters.
            self._characters = new_characters

        if level == 3:
            # Step 3: Complete the sequence of scenes.
            # Get the text of the current scenes.
            text_scenes = self._scenes.to_string()
            # Strip away any trailing newlines.
            text_scenes = strip_remove_end(text_scenes, END_MARKER)
            # Build the prompt by combining the prompt for scenes with the
            # current scenes.
            prompt = self.prompts['scenes'] + text_scenes
            # Generate the new scenes.
            text = self.generate_text(
                generation_prompt=prompt,
                seed=seed)
            # Create a new Scenes object from the generated text and the
            # current scenes.
            new_scenes = Scenes.from_string(text_scenes + text)
            # Compute the diff between the old and new scenes.
            prompt_diff = diff_prompt_change_scenes(self._scenes.scenes,
                                                    new_scenes.scenes)
            # Set the scenes to the new scenes.
            self._scenes = new_scenes

        if level == 5:
            # Step 5: Complete the dialog of a given scene.
            # Get the dialogs.
            dialogs = self._dialogs
            # Get the number of scenes.
            num_scenes = len(self._scenes.scenes)
            # Make sure there are enough dialogs.
            while len(self._dialogs) < num_scenes:
                self._dialogs.append('')
            # Make sure there are enough prompts for dialogs.
            while len(self.prompts['dialogs']) < num_scenes:
                self.prompts['dialogs'].append('')
            # If entity is given, generate the dialog for the given scene.
            if entity >= 0 and entity < num_scenes:
                # Build the prompt by combining the prompt for dialogs with the
                # current dialog.
                prompt = (self.prompts['dialogs'][entity] + self._dialogs[entity])
                # Generate the new dialog.
                text = self.generate_text(
                    generation_prompt=prompt,
                    seed=seed)
                # Create a new string from the generated text and the current
                # dialog.
                new_dialog = self._dialogs[entity] + text
                # Compute the diff between the old and new dialog.
                prompt_diff = diff_prompt_change_str(self._dialogs[entity], new_dialog)
                # Set the dialog to the new dialog.
                self._dialogs[entity] = new_dialog

        # Keep track of each rewrite intervention.
        if prompt_diff is not None and len(prompt_diff) > 0:
            timestamp = time.time()
            self.interventions[timestamp] = 'COMPLETE ' + self.level_names[level]
            if entity:
                self.interventions[timestamp] += ' ' + str(entity)
            self.interventions[timestamp] += prompt_diff


# class StoryGenerator:
#   """Generate a story from the provided storyline, using the client provided."""

#   level_names = ('storyline', 'title', 'characters', 'scenes', 'places',
#                  'dialogs')

#   def __init__(
#       self,
#       storyline: str,
#       prefixes: Dict[str, str],
#       max_paragraph_length: int = 1024,
#       max_paragraph_length_characters: int = (MAX_PARAGRAPH_LENGTH_CHARACTERS),
#       max_paragraph_length_scenes: int = (MAX_PARAGRAPH_LENGTH_SCENES),
#       num_samples: int = 1,
#       client: Optional[LanguageAPI] = None,
#       filter: Optional[FilterAPI] = None):
#     self._prefixes = prefixes
#     self._max_paragraph_length = max_paragraph_length
#     self._max_paragraph_length_characters = max_paragraph_length_characters
#     self._max_paragraph_length_scenes = max_paragraph_length_scenes
#     self._num_samples = num_samples
#     self._client = client
#     self._filter = filter
#     print('STORY GENERATOR WAS INITIALIZED!')
#     # Prompts and outputs of the hierarchical generator are organised in levels.
#     self.prompts = {
#         'title': '',
#         'characters': '',
#         'scenes': '',
#         'places': {
#             '': ''
#         },
#         'dialogs': ['']
#     }
#     self._title = Title('')
#     self._characters = Characters({'': ''})
#     self._scenes = Scenes([Scene('', '', '')])
#     self._places = {'': Place('', '')}
#     self._dialogs = ['']

#     # History of interventions.
#     self.interventions = {}
#     self._set_storyline(storyline)

#   def _set_storyline(self, storyline: str):
#     """Set storyline and initialise the outputs of the generator."""
#     self._level = 0

#     # Add period to the end of the storyline, unless there is already one there.
#     if storyline.find('.') == -1:
#       storyline = storyline + '.'
#     self._storyline = storyline

#     # Keep track of each storyline intervention.
#     timestamp = time.time()
#     self.interventions[timestamp] = 'STORYLINE\n' + storyline

#   @property
#   def seed(self):
#     return self._client.seed

#   @property
#   def title(self) -> Title:
#     """Return the title."""
#     return self._title

#   @property
#   def characters(self) -> Characters:
#     """Return the characters."""
#     return self._characters

#   @property
#   def scenes(self) -> Scenes:
#     """Return the title."""
#     return self._scenes

#   @property
#   def places(self) -> Dict[str, Place]:
#     """Return the places."""
#     return self._places

#   @property
#   def dialogs(self) -> List[str]:
#     """Return the dialogs."""
#     return self._dialogs

#   def title_str(self) -> str:
#     """Return the title as a string."""
#     return self._title.title

#   def num_scenes(self) -> int:
#     """Return the number of scenes."""
#     return self._scenes.num_scenes()
  
#   def reasign_title(self, updated_text):
#     self._title = Title.from_string(TITLE_ELEMENT + updated_text)

#   def reasign_chars(self, updated_text):
#     self._characters = Characters.from_string(updated_text)

#   def reasign_scene(self, idx_of_element, updated_text, new_place = None, new_plot_element = None, new_beat_element = None):
#     copy_of_scene =  self._scenes[0][idx_of_element]
#     list_of_all_items = [new_place, new_plot_element, new_beat_element]

#     if updated_text is not None and new_place is None and new_plot_element is None and new_beat_element is None:
#         places_list = extract_elements(updated_text, PLACE_ELEMENT, PLOT_ELEMENT)
#         plots_list = extract_elements(updated_text, PLOT_ELEMENT, BEAT_ELEMENT)
#         updated_beats_list = extract_elements(updated_text, BEAT_ELEMENT, '\n')

#         # Get the number of complete scenes.
#         num_complete_scenes = min([len(places_list), len(plots_list), len(updated_beats_list)])

#         if num_complete_scenes < max([len(places_list), len(plots_list), len(updated_beats_list)]):
#             places_list, plots_list, updated_beats_list = parse_places_plot_beats(updated_text)
        
#             num_complete_scenes = min[len(places_list), len(plots_list), len(updated_beats_list)]
#             if num_complete_scenes == 0:
#                 places_list, plots_list, updated_beats_list = parse_narrative_elements(updated_text)
            
#         updated_scene_element = Scene(Place.format_name(places_list[idx_of_element]), plots_list[idx_of_element], updated_beats_list[idx_of_element])
#         self._scenes[0][idx_of_element] = updated_scene_element
    
#     elif updated_text is None:
#         for ind, elem in enumerate(list_of_all_items):
#             if ind == 0:
#                 if elem is None:
#                     elem = copy_of_scene.place
#                     list_of_all_items[ind] = elem
#             if ind == 1:
#                 if elem is None:
#                     elem = copy_of_scene.plot_element
#                     list_of_all_items[ind] = elem

#             if ind == 2:
#                 if elem is None:
#                     elem = copy_of_scene.beat
#                     list_of_all_items[ind] = elem
        
#         print(list_of_all_items)
#         updated_scene_element = Scene(Place.format_name(list_of_all_items[0]), list_of_all_items[1], list_of_all_items[2])
#         self._scenes[0][idx_of_element]  = updated_scene_element


#   def reasign_new_place(self, place_name, new_desc):
#     updated_elem = Place(name = place_name, description=new_desc)
#     self._places.pop(place_name)
#     self._places[place_name] = updated_elem

#   def reasign_dialog(self, idx_of_element, new_dialog):
#     self._dialogs[idx_of_element] = new_dialog
    

  
#                level=0,
#                seed=None,
#                entity=None,
#                sample_length=SAMPLE_LENGTH):
#     if level < 0 or level >= len(self.level_names):
#       raise ValueError('Invalid level encountered on step.')
#     prompt_diff = None

#     if level == 2:
#       # Step 2: Complete the characters.
#       text_characters = self._characters.to_string()
#       text_characters = strip_remove_end(text_characters)
#       prompt = self.prompts['characters'] + text_characters
#       text = generate_text(
#           generation_prompt=prompt,
#           client=self._client,
#           filter=self._filter,
#           sample_length=sample_length,
#           max_paragraph_length=sample_length,
#           seed=seed,
#           num_samples=1)
#       new_characters = Characters.from_string(text_characters + text)
#       prompt_diff = diff_prompt_change_dict(
#           self._characters.character_descriptions,
#           new_characters.character_descriptions)
#       self._characters = new_characters

#     if level == 3:
#       # Step 3: Complete the sequence of scenes.
#       text_scenes = self._scenes.to_string()
#       text_scenes = strip_remove_end(text_scenes)
#       prompt = self.prompts['scenes'] + text_scenes
#       text = generate_text(
#           generation_prompt=prompt,
#           client=self._client,
#           filter=self._filter,
#           sample_length=sample_length,
#           max_paragraph_length=sample_length,
#           seed=seed,
#           num_samples=1)
#       new_scenes = Scenes.from_string(text_scenes + text)
#       prompt_diff = diff_prompt_change_scenes(self._scenes.scenes,
#                                               new_scenes.scenes)
#       self._scenes = new_scenes

#     if level == 5:
#       # Step 5: Complete the dialog of a given scene.
#       dialogs = self._dialogs
#       num_scenes = len(self._scenes.scenes)
#       while len(self._dialogs) < num_scenes:
#         self._dialogs.append('')
#       while len(self.prompts['dialogs']) < num_scenes:
#         self.prompts['dialogs'].append('')
#       if entity >= 0 and entity < num_scenes:
#         prompt = (self.prompts['dialogs'][entity] + self._dialogs[entity])
#         text = generate_text(
#             generation_prompt=prompt,
#             client=self._client,
#             filter=self._filter,
#             sample_length=sample_length,
#             max_paragraph_length=sample_length,
#             seed=seed,
#             num_samples=1)
#         new_dialog = self._dialogs[entity] + text
#         prompt_diff = diff_prompt_change_str(self._dialogs[entity], new_dialog)
#         self._dialogs[entity] = new_dialog

#     # Keep track of each rewrite intervention.
#     if prompt_diff is not None and len(prompt_diff) > 0:
#       timestamp = time.time()
#       self.interventions[timestamp] = 'COMPLETE ' + self.level_names[level]
#       if entity:
#         self.interventions[timestamp] class StoryGenerator:
#   """Generate a story from the provided storyline, using the client provided."""

#   level_names = ('storyline', 'title', 'characters', 'scenes', 'places',
#                  'dialogs')

#   def __init__(
#       self,
#       storyline: str,
#       prefixes: Dict[str, str],
#       max_paragraph_length: int = 1024,
#       max_paragraph_length_characters: int = (MAX_PARAGRAPH_LENGTH_CHARACTERS),
#       max_paragraph_length_scenes: int = (MAX_PARAGRAPH_LENGTH_SCENES),
#       num_samples: int = 1,
#       client: Optional[LanguageAPI] = None,
#       filter: Optional[FilterAPI] = None):
#     self._prefixes = prefixes
#     self._max_paragraph_length = max_paragraph_length
#     self._max_paragraph_length_characters = max_paragraph_length_characters
#     self._max_paragraph_length_scenes = max_paragraph_length_scenes
#     self._num_samples = num_samples
#     self._client = client
#     self._filter = filter
#     print('STORY GENERATOR WAS INITIALIZED!')
#     # Prompts and outputs of the hierarchical generator are organised in levels.
#     self.prompts = {
#         'title': '',
#         'characters': '',
#         'scenes': '',
#         'places': {
#             '': ''
#         },
#         'dialogs': ['']
#     }
#     self._title = Title('')
#     self._characters = Characters({'': ''})
#     self._scenes = Scenes([Scene('', '', '')])
#     self._places = {'': Place('', '')}
#     self._dialogs = ['']

#     # History of interventions.
#     self.interventions = {}
#     self._set_storyline(storyline)

#   def _set_storyline(self, storyline: str):
#     """Set storyline and initialise the outputs of the generator."""
#     self._level = 0

#     # Add period to the end of the storyline, unless there is already one there.
#     if storyline.find('.') == -1:
#       storyline = storyline + '.'
#     self._storyline = storyline

#     # Keep track of each storyline intervention.
#     timestamp = time.time()
#     self.interventions[timestamp] = 'STORYLINE\n' + storyline

#   @property
#   def seed(self):
#     return self._client.seed

#   @property
#   def title(self) -> Title:
#     """Return the title."""
#     return self._title

#   @property
#   def characters(self) -> Characters:
#     """Return the characters."""
#     return self._characters

#   @property
#   def scenes(self) -> Scenes:
#     """Return the title."""
#     return self._scenes

#   @property
#   def places(self) -> Dict[str, Place]:
#     """Return the places."""
#     return self._places

#   @property
#   def dialogs(self) -> List[str]:
#     """Return the dialogs."""
#     return self._dialogs

#   def title_str(self) -> str:
#     """Return the title as a string."""
#     return self._title.title

#   def num_scenes(self) -> int:
#     """Return the number of scenes."""
#     return self._scenes.num_scenes()
  
#   def reasign_title(self, updated_text):
#     self._title = Title.from_string(TITLE_ELEMENT + updated_text)

#   def reasign_chars(self, updated_text):
#     self._characters = Characters.from_string(updated_text)

#   def reasign_scene(self, idx_of_element, updated_text, new_place = None, new_plot_element = None, new_beat_element = None):
#     copy_of_scene =  self._scenes[0][idx_of_element]
#     list_of_all_items = [new_place, new_plot_element, new_beat_element]

#     if updated_text is not None and new_place is None and new_plot_element is None and new_beat_element is None:
#         places_list = extract_elements(updated_text, PLACE_ELEMENT, PLOT_ELEMENT)
#         plots_list = extract_elements(updated_text, PLOT_ELEMENT, BEAT_ELEMENT)
#         updated_beats_list = extract_elements(updated_text, BEAT_ELEMENT, '\n')

#         # Get the number of complete scenes.
#         num_complete_scenes = min([len(places_list), len(plots_list), len(updated_beats_list)])

#         if num_complete_scenes < max([len(places_list), len(plots_list), len(updated_beats_list)]):
#             places_list, plots_list, updated_beats_list = parse_places_plot_beats(updated_text)
        
#             num_complete_scenes = min[len(places_list), len(plots_list), len(updated_beats_list)]
#             if num_complete_scenes == 0:
#                 places_list, plots_list, updated_beats_list = parse_narrative_elements(updated_text)
            
#         updated_scene_element = Scene(Place.format_name(places_list[idx_of_element]), plots_list[idx_of_element], updated_beats_list[idx_of_element])
#         self._scenes[0][idx_of_element] = updated_scene_element
    
#     elif updated_text is None:
#         for ind, elem in enumerate(list_of_all_items):
#             if ind == 0:
#                 if elem is None:
#                     elem = copy_of_scene.place
#                     list_of_all_items[ind] = elem
#             if ind == 1:
#                 if elem is None:
#                     elem = copy_of_scene.plot_element
#                     list_of_all_items[ind] = elem

#             if ind == 2:
#                 if elem is None:
#                     elem = copy_of_scene.beat
#                     list_of_all_items[ind] = elem
        
#         print(list_of_all_items)
#         updated_scene_element = Scene(Place.format_name(list_of_all_items[0]), list_of_all_items[1], list_of_all_items[2])
#         self._scenes[0][idx_of_element]  = updated_scene_element


#   def reasign_new_place(self, place_name, new_desc):
#     updated_elem = Place(name = place_name, description=new_desc)
#     self._places.pop(place_name)
#     self._places[place_name] = updated_elem

#   def reasign_dialog(self, idx_of_element, new_dialog):
#     self._dialogs[idx_of_element] = new_dialog
    

  
#                level=0,
#                seed=None,
#                entity=None,
#                sample_length=SAMPLE_LENGTH):
#     if level < 0 or level >= len(self.level_names):
#       raise ValueError('Invalid level encountered on step.')
#     prompt_diff = None

#     if level == 2:
#       # Step 2: Complete the characters.
#       text_characters = self._characters.to_string()
#       text_characters = strip_remove_end(text_characters)
#       prompt = self.prompts['characters'] + text_characters
#       text = generate_text(
#           generation_prompt=prompt,
#           client=self._client,
#           filter=self._filter,
#           sample_length=sample_length,
#           max_paragraph_length=sample_length,
#           seed=seed,
#           num_samples=1)
#       new_characters = Characters.from_string(text_characters + text)
#       prompt_diff = diff_prompt_change_dict(
#           self._characters.character_descriptions,
#           new_characters.character_descriptions)
#       self._characters = new_characters

#     if level == 3:
#       # Step 3: Complete the sequence of scenes.
#       text_scenes = self._scenes.to_string()
#       text_scenes = strip_remove_end(text_scenes)
#       prompt = self.prompts['scenes'] + text_scenes
#       text = generate_text(
#           generation_prompt=prompt,
#           client=self._client,
#           filter=self._filter,
#           sample_length=sample_length,
#           max_paragraph_length=sample_length,
#           seed=seed,
#           num_samples=1)
#       new_scenes = Scenes.from_string(text_scenes + text)
#       prompt_diff = diff_prompt_change_scenes(self._scenes.scenes,
#                                               new_scenes.scenes)
#       self._scenes = new_scenes

#     if level == 5:
#       # Step 5: Complete the dialog of a given scene.
#       dialogs = self._dialogs
#       num_scenes = len(self._scenes.scenes)
#       while len(self._dialogs) < num_scenes:
#         self._dialogs.append('')
#       while len(self.prompts['dialogs']) < num_scenes:
#         self.prompts['dialogs'].append('')
#       if entity >= 0 and entity < num_scenes:
#         prompt = (self.prompts['dialogs'][entity] + self._dialogs[entity])
#         text = generate_text(
#             generation_prompt=prompt,
#             client=self._client,
#             filter=self._filter,
#             sample_length=sample_length,
#             max_paragraph_length=sample_length,
#             seed=seed,
#             num_samples=1)
#         new_dialog = self._dialogs[entity] + text
#         prompt_diff = diff_prompt_change_str(self._dialogs[entity], new_dialog)
#         self._dialogs[entity] = new_dialog

#     # Keep track of each rewrite intervention.
#     if prompt_diff is not None and len(prompt_diff) > 0:
#       timestamp = time.time()
#       self.interventions[timestamp] = 'COMPLETE ' + self.level_names[level]
#       if entity:
#         self.interventions[timestamp] += ' ' + str(entity)
#       self.interventions[timestamp] += prompt_diff+= ' ' + str(entity)
#       self.interventions[timestamp] += prompt_diff