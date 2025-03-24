import openai
from dotenv import load_dotenv
from openai import OpenAI

from HyperParameters import *
# from examples import *



load_dotenv()


def create_prompt_for_NER(text_for_processing):

    prompt = """
    Convert the following dialogue text into a list of dictionaries. Each dictionary should represent a character, with the character's name as the key and another dictionary as the value. In this inner dictionary, the keys are sequence numbers based on the overall order of dialogue lines, and the values are the character's dialogue lines. 
    Exclude narrative text, stage directions, and any non-dialogue content.
    !!! BE CAREFUL WITH NUMBERS, YOU TEND TO CONFUSE THEM !!!!
    For example, given the dialogue:

    Sasha: Hello, my friend!
    Sasha: How are you doing?
    Oleg: Hi. Nothing special, just like everyone else.

    The output should be:

    {'Sasha': {1: 'Hello, my friend!', 2: 'How are you doing?'}},
    {'Oleg': {3: 'Hi. Nothing special, just like everyone else.'}}

    Return the results as a plain text.    
    Text for processing: 
    """
    prompt = prompt + text_for_processing
    return prompt


def create_prompt_simple(text_for_processing):

    prompt = f"""
    Clean the following text from narrative text, stage directions, and any non-dialogue content, and emotion-action tags such as (whispering), (smirking), (from above), (turning) and so on. Preserve the order and return it in a following format:
    _____
    character_name
    replica
    ....
    ______
    Text for processing: {text_for_processing}
    """
    return prompt



def send_request(prompt, temperature =  0, model_name = 'gpt-4o', max_tokens = 1024):
    message_log = [
      {
          "role":"user",
          "content": prompt,
      },
  ]
    if 'gpt' in model_name:
        client = OpenAI()

        response = client.chat.completions.create(
                model=model_name,
                messages=message_log,
                max_tokens = max_tokens,
                temperature=0,
                presence_penalty=0,
                frequency_penalty=0
            )
        output = response.choices[0].message.content
    else:
        raise NotImplementedError

    return output


def edit_the_result_with_prompt(prompt, text_to_edit, model_name = 'gpt-4o', max_tokens = 4096, temperature = 0, input_type = 'title'):
    
    if input_type == 'title' or input_type == 'plain_text':
        tmplt = "You are a helpful editor assistant. Replicate the format that is used in the input. As output you need to return only generated output.\n"  
    else:
        tmplt = "You are a helpful editor assistant. Replicate the format that is used in the input. In output you need to combine the input with generated output.\n"
    
    prompt = tmplt + prompt
    prompt += text_to_edit

    print(prompt)    
    output = send_request(prompt, temperature =temperature, model_name='gpt-4o', max_tokens = max_tokens)

    return output



def add_custom_chars_to_prompt(prefixes:dict, custom_data:str):
    prefixes['CHARACTERS_PROMPT_CUSTOM'] = """
    Here is an example of a logline and a list of characters.

    """ + LOGLINE_MARKER + """James finds a well in his backyard that is haunted by the ghost of Sam.

    """ + CHARACTER_MARKER + """ James """ + DESCRIPTION_MARKER + """ James is twenty-six, serious about health and wellness and optimistic. """ + STOP_MARKER + """
    """ + CHARACTER_MARKER + """ Sam """ + DESCRIPTION_MARKER + """ Sam fell down the well when he was 12, and was never heard from again. Sam is now a ghost. """ + STOP_MARKER + """
    """ + END_MARKER + """

    Example 2.

    """ + LOGLINE_MARKER + """Morgan adopts a new cat, Misterio, who sets a curse on anyone that pets them.

    """ + CHARACTER_MARKER + """ Morgan """ + DESCRIPTION_MARKER + """ Morgan is booksmart and popular; they are trusting but also have been known to hold a grudge. """ + STOP_MARKER + """
    """ + CHARACTER_MARKER + """ Misterio """ + DESCRIPTION_MARKER + """ Misterio is a beautiul black cat, it is of uncertain age; it has several gray whiskers that make it look wise and beyond its years.  """ + STOP_MARKER + """
    """ + END_MARKER + """

    Example 3.

    """ + LOGLINE_MARKER + """Mr. Dorbenson finds a book at a garage sale that tells the story of his own life. And it ends in a murder!

    """ + CHARACTER_MARKER + """ Mr. Glen Dorbenson """ + DESCRIPTION_MARKER + """ Mr. Glen Dorbenson frequents markets and garage sales always looking for a bargain. He is lonely and isolated and looking for his meaning in life. """ + STOP_MARKER + """
    """ + END_MARKER + """

    Using the examples above, provided initial data about characters and the following logline, complete the list of characters.

    """ + custom_data + '\n' + LOGLINE_MARKER

    return prefixes

    
    

def assign_items_with_names(list_with_names, list_of_references):
    """
    Create a dict of references that will be usd for generation.
    """
    ref_dict = {}
    for name, ref in zip(list_with_names, list_of_references):
        ref_dict[name] = ref

    return ref_dict


def create_prompt_for_rec(rec_request, data):
    prompt = f"""You are a helpful editor assistant. You need to recommendations according to request using provided script summart.\n
        Request: {rec_request}.\n
        Summary: {data}. \n
        """
    
    return prompt


def parse_output(list_of_names, text_for_processing):
    text_splitted = text_for_processing.split('\n')
    # print(text_splitted)
    # print(len(text_splitted))
    list_of_elements = []
    for i in range(len(text_splitted)-2):
        if text_splitted[i] in list_of_names and text_splitted[i+1] != '' and (text_splitted[i+2] == '' or text_splitted[i+2] == '_____'):
            list_to_add = [text_splitted[i], text_splitted[i+1]]
            list_of_elements.append(list_to_add)
    gen_output_dict = {}
    for name in list_of_names:
        gen_output_dict[name] = None
        output_dict = {}
        for i, item in enumerate(list_of_elements):
            if item[0] == name:
                output_dict[i+1] = item[1]
        gen_output_dict[name] = output_dict


    print(gen_output_dict)
    return gen_output_dict


def map_the_voices_and_pictures(map_dict, audio_dictionary):
    """
    map_dict : dictionary of the following format: key = name of the character, value  = path to the portrit picture.
    audio_dictionary : output dictionary of the audio generation models.
    """
    mod_dictionary = {}
    for key, value in audio_dictionary.items():
        mod_dictionary[key] = [map_dict[key], value]

    return mod_dictionary



def parse_toxic_rating(output_of_the_model):
    output = output_of_the_model.strip()

    numbers_str = output.split(',')

    numbers = list(map(float, numbers_str))

    print(numbers)

    return numbers