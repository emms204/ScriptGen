import os
import streamlit as st
from TextGen.dramatron.DramaTronCore import *
from src.TextGen.HyperParameters import *
from src.TextGen.utils import *
from src.TextGen.TextGen import *
from TextGen.examples import custom_prefixes
from TextGen.examples import config
import pickle
from src.ImageProcessing.ImageProcessor import ImageGenerator, Img2Text
import io
from PIL import Image

def check_empty_dialog_list(list_of_dd):
    flag_list = []

    try:
        for item in list_of_dd:
            if item['raw_text'] is None:
                flag_list.append(True)
            else:
                flag_list.append(False)
        if flag_list == []:
            return None
        return sum(flag_list)
    except:
        for item in list_of_dd:
            if item != '':
                flag_list.append(False)
            else:
                flag_list.append(True)

        if flag_list == []:
            return None
        return sum(flag_list)
    
def save_images(output_path, dict_with_images):
    counter = 0
    for key, list_elem in dict_with_images.items():
        out_path = os.path.join(output_path, str(counter))
        list_elem[0].save(f'{out_path}.jpg')
        with open(f'{out_path}.txt', 'w') as desc_file:
            desc_file.write(list_elem[1])
        counter+=1


base_dir = 'generated_images/storyboards'



if 'model_name' not in st.session_state.keys():
    st.session_state.model_name = None
    st.session_state.model_name_selected = None
    st.session_state.client = None
    st.session_state.generator = None
    st.session_state.is_loaded = False
    st.session_state.prefixes  = custom_prefixes
    st.session_state.logline = None
    st.session_state.titles_generated = False
    st.session_state.characters_generated = False
    st.session_state.scenes_generated = False
    st.session_state.data_titles =  None 
    st.session_state.data_chars = None
    st.session_state.data_scenes = None
    st.session_state.place_names = None
    st.session_state.data_places = None
    st.session_state.list_of_data_dialogs = []
    st.session_state.story = None

    st.session_state.title_gen = None
    st.session_state.title_regen = None
    st.session_state.title_switcher = False

    st.session_state.char_gen = None
    st.session_state.char_regen = None
    st.session_state.char_switcher = False

    st.session_state.scene_gen = None
    st.session_state.scene_regen = None
    st.session_state.scene_switcher = False

    st.session_state.places_gen = None
    st.session_state.places_regen = None
    st.session_state.places_switcher = None
    
    st.session_state.gen_dialogs = None
    st.session_state.regen_dialogs = None
    st.session_state.dialog_switcher = None

    st.session_state.num_scenes = None
    st.session_state.list_of_data_dialogs_filled = []
    st.session_state.to_parse_and_save = False

if 'edit_title' not in st.session_state.keys():
    st.session_state.edit_title = False
    st.session_state.edit_title_choice = None
    st.session_state.edit_title_text = None
    st.session_state.edit_title_prompt = None
    st.session_state.reassign_title = False

    st.session_state.edit_characters = False
    st.session_state.edit_characters_choice = None
    st.session_state.edit_characters_text = None
    st.session_state.edit_characters_prompt = None
    st.session_state.reassign_characters = False

    st.session_state.edit_scenes = False
    st.session_state.edit_scenes_choice = None
    st.session_state.edit_scenes_text = None
    st.session_state.edit_scenes_prompt = None
    st.session_state.reassign_scenes = False

    st.session_state.edit_places = False
    st.session_state.edit_places_choice = None
    st.session_state.edit_places_text = None
    st.session_state.edit_places_prompt = None
    st.session_state.reassign_places = False

    st.session_state.edit_dialog = False
    st.session_state.edit_dialog_choice = None
    st.session_state.edit_dialog_text = None
    st.session_state.edit_dialog_prompt = None
    st.session_state.reassign_dialog = False
    st.session_state.prompting_approach = None

if 'analyze_toxicity' not in st.session_state.keys():
    st.session_state.analyze_toxicity = None
    st.session_state.use_dialog_sequence = None
    st.session_state.generate_images = None
    st.session_state.img_gen_model_type = None
    st.session_state.img_gen_model_name = None
    st.session_state.enter_model_name = None
    st.session_state.initial_char_images = None
    st.session_state.initial_places_images = None
    st.session_state.upload = None
    
    st.session_state.initial_scenes_data = None
    st.session_state.initial_places_data = None
    
    st.session_state.initial_scenes_data_confirm = None
    st.session_state.initial_places_data_confirm = False

open_source_llms = ['LongWriter-glm4-9b', 'LongWriter-llama3.1-8b', 'Phi-3.5-MoE']


if 'use_predefined_chars' not in st.session_state.keys():
    st.session_state.use_predefined_chars = False
    st.session_state.use_predefined_scenes = False
    st.session_state.use_predefined_places = False
    st.session_state.initial_char_data = None
    st.session_state.initial_char_data_confirm = False

st.title('Pipeline demonstration.')

if st.session_state.model_name_selected == False or st.session_state.model_name_selected == None:
    st.session_state.model_name = st.selectbox('Choose the model type:', ['gpt', 'sonnet', 'gemini', 'LongWriter-glm4-9b', 'LongWriter-llama3.1-8b', 'Phi-3.5-MoE'], index = 0,)
    st.session_state.prompting_approach = st.selectbox('Choose prompting approach.', ['DramaTron', 'Custom'], index = 0)
    st.session_state.logline = st.text_input('Write the log line.', value = "Epic story about two knights lost in the enchanted woods during a quest.")
    st.session_state.analyze_toxicity = st.checkbox('Use toxicity filter.')
    st.session_state.use_dialog_sequence = st.checkbox('Use dialog sequence.')
    st.session_state.generate_images = st.checkbox('Generate images of characters and places.')
    if st.session_state.generate_images == True:
        st.session_state.img_gen_model_type = st.selectbox('Choose the image generator model type:', ['Flux', 'StableDiffusion'])
        if st.session_state.img_gen_model_type == 'Flux':
            st.session_state.img_gen_model_name = st.selectbox('Choose the model name', ['dev', 'schnell'])
        
        elif st.session_state.img_gen_model_type == 'StableDiffusion':
            st.session_state.enter_model_name = st.selectbox('Enter the model name or choose from predefined.')
            if st.session_state.enter_model_name == True:
                st.session_state.img_gen_model_name = st.text_input('Enter the name of the HF checkpoint.', value = "stabilityai/stable-diffusion-xl-base-1.0")
            elif st.session_state.enter_model_name == False:
                st.session_state.img_gen_model_name = st.selectbox('Choose the model name', ['stable-diffusion-xl'])

            
    st.session_state.model_name_selected = st.button('Confirm your choise')



if st.session_state.model_name_selected == True and st.session_state.is_loaded == False:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if st.session_state.model_name in open_source_llms:
        if device == 'cpu':
            raise AssertionError('For proper work GPU should be available.')
        
    if st.session_state.prompting_approach == 'DramaTron':
        st.session_state.client  = DramatronHandler(
            model_type=st.session_state.model_name,
            seed=DEFAULT_SEED,
            sample_length=config['sample_length'],
            max_retries=config['max_retries'],
            config_sampling=config['sampling'],
            timeout=TIMEOUT)
        
        st.session_state.generator = StoryGenerator(
            storyline=st.session_state.logline,
            prefixes=st.session_state.prefixes,
            max_paragraph_length=config['max_paragraph_length'],
            client=st.session_state.client,
            filter=filter)
        
    elif st.session_state.prompting_approach == 'Custom':
        st.session_state.client  = DramatronHandler(
            model_type=st.session_state.model_name,
            seed=DEFAULT_SEED,
            sample_length=config['sample_length'],
            max_retries=config['max_retries'],
            config_sampling=config['sampling'],
            timeout=TIMEOUT)
        
        st.session_state.generator = StoryGenerator(
            storyline=st.session_state.logline,
            prefixes=st.session_state.prefixes,
            max_paragraph_length=config['max_paragraph_length'],
            client=st.session_state.client,
            filter=filter)
        
        st.session_state.story_generator = StoryGeneratorCustom(st.session_state.client, st.session_state.generator, logline = st.session_state.logline, analyze_toxicity = st.session_state.analyze_toxicity, dialog_sequence = st.session_state.use_dialog_sequence, config = config)
    
    if st.session_state.generate_images == True:
        if device == 'cpu':
            st.write('Images will not be generated due to GPU absence.')
        if device == 'cuda':
            st.session_state.image_generator = ImageGenerator(st.session_state.img_gen_model_type, st.session_state.img_gen_model_name, 'gpt-4o', temperature=0.5)
            st.write('Image generator was loaded!')    
            
    st.session_state.img2text = Img2Text(st.session_state.model_name)   

    st.write('Generator and client was loaded.')

    st.session_state.is_loaded = True

    st.session_state.data_titles = {"text": "", "text_area": None, "seed": st.session_state.generator.seed - 1}
    st.session_state.data_chars = {"text": "", "text_area": None, "seed": st.session_state.generator.seed - 1, "history": GenerationHistory(), "lock": False}
    st.session_state.data_scenes = {"text": "", "text_area": None, "seed": st.session_state.generator.seed - 1,
               "history": GenerationHistory(), "lock": False}
    st.session_state.place_names = list(set([scene.place for scene in st.session_state.generator.scenes[0]]))
    st.session_state.place_descriptions  = {place_name: Place(place_name, '') for place_name in st.session_state.place_names}
    st.session_state.data_places = {"descriptions": st.session_state.place_descriptions, "text_area": {}, "seed": st.session_state.generator.seed - 1}


##################################################################################################
if st.session_state.is_loaded == True:
    st.write("<h4>Generate Title</h4>", unsafe_allow_html=True)
    
    if st.session_state.data_titles['text'] == '':
        st.session_state.title_gen = st.button('Generate titles')

    if st.session_state.data_titles['text'] != '':
        st.session_state.title_regen = st.button('Regenerate title')

    if st.session_state.title_gen == True:
        if st.session_state.prompting_approach == 'DramaTron':
            fun_generate_title(st.session_state.data_titles, st.session_state.generator)
        
        elif st.session_state.prompting_approach == 'Custom':
            st.session_state.story_generator.generate('title', 4096)
            st.session_state.data_titles['text'] = st.session_state.story_generator.title.title
    
    if st.session_state.title_regen == True:
        if st.session_state.prompting_approach == 'DramaTron':
            fun_generate_title(st.session_state.data_titles, st.session_state.generator)
        
        elif st.session_state.prompting_approach == 'Custom':
            st.session_state.story_generator.generate('title', 4096)
            st.session_state.data_titles['text'] = st.session_state.story_generator.title.title

    if st.session_state.data_titles is not None and st.session_state.data_titles['text'] != "":
        st.text_area(label = 'Title', value = st.session_state.data_titles['text'])
        st.session_state.title_gen = False
        if st.session_state.edit_title == False:
            st.session_state.edit_title = st.button('Edit title')
        if st.session_state.edit_title == True:
            st.session_state.edit_title_choice = st.radio('How to edit:', ['Manual','LLM'])

        if st.session_state.edit_title_choice is not None and st.session_state.edit_title_choice == 'Manual':
            st.session_state.edit_title_text = st.text_input(label = 'Input the text')
            st.session_state.reassign_title = st.button('Reassign the title')
            if st.session_state.edit_title_text is not None and st.session_state.edit_title_text != '' and st.session_state.reassign_title == True:
                # st.session_state.generator.reasign_title(st.session_state.edit_title_text)
                st.session_state.generator._title = st.session_state.edit_title_text
                st.write('Title was updated. generator._title is below.')
                st.text(st.session_state.generator._title) 
        
        elif st.session_state.edit_title_choice is not None and st.session_state.edit_title_choice == 'LLM':
            st.session_state.edit_title_prompt = st.text_input(label = 'Input the prompt for change')
            if st.session_state.edit_title_prompt is not None and st.session_state.edit_title_prompt != '' and st.session_state.reassign_title == False:
                output_text = edit_the_result_with_prompt(prompt = st.session_state.edit_title_prompt,
                                        text_to_edit=st.session_state.generator._title.to_string(), 
                                          temperature=0.4)
                st.session_state.edit_title_text = output_text
                st.session_state.reassign_title = st.button('Reassign the title')
                if st.session_state.edit_title_text is not None and st.session_state.edit_title_text != '' and st.session_state.reassign_title == True:
                    # st.session_state.generator.reasign_title(st.session_state.edit_title_text)
                    st.session_state.generator._title = st.session_state.edit_title_text
                    st.write('Title was updated. generator._title is below.')
                    st.text(st.session_state.generator._title) 

##################################################################################################
    st.write("<h4>Generate Characters</h4>", unsafe_allow_html=True)
    
    if st.session_state.data_chars['text'] == '':
        use_predefined_chars = st.checkbox(label= 'Use predefined characters?', value = False)
        st.session_state.initial_char_images = st.checkbox(label = 'Use initial characters data as images')
        if st.session_state.initial_char_images == True:
            use_predefined_chars = True
        
        if use_predefined_chars == True:
            if st.session_state.initial_char_images == False:
                if st.session_state.initial_char_data_confirm == False:
                    st.session_state.initial_char_data = st.text_input(label = 'Initial characters description:')
                    st.session_state.initial_char_data_confirm = st.button('Confirm the prompt.')
            elif st.session_state.initial_char_images == True:
                temp_img_path = 'temp.jpg'
                if st.session_state.initial_char_data_confirm == False:
                    st.session_state.files = st.file_uploader('Upload the images', accept_multiple_files=True)
                    st.session_state.upload = st.button(label = 'Upload images')

                    if st.session_state.upload == True:
                        st.session_state.initial_char_data = ''
                        list_of_output_results = []
                        for ind, item in  enumerate(st.session_state.files):
                            byte_io = io.BytesIO(item.read())
                            image = Image.open(byte_io)
                            image.save(temp_img_path)
                            conv_result = st.session_state.img2text.convert(temp_img_path)
                            list_of_output_results.append(f'Character {ind} - Description: {conv_result}')
                        st.session_state.initial_char_data = '\n\n'.join(list_of_output_results)               
                        print(st.session_state.initial_char_data)
                        st.session_state.initial_char_data_confirm = st.button('Confirm the description')
        st.session_state.char_gen = st.button('Generate characters')

    if st.session_state.data_chars['text'] != '':
        st.session_state.char_regen = st.button('Regenerate characters')

    print(f"Flag: {[st.session_state.char_gen == True, st.session_state.initial_char_data is not None, st.session_state.initial_char_data != '', st.session_state.initial_char_data_confirm == True]}")
    if st.session_state.char_gen == True \
    and st.session_state.initial_char_data is not None \
    and st.session_state.initial_char_data != '' :
        
        if st.session_state.prompting_approach == 'DramaTron':
            print('Drama Tron method was called.')
            fun_generate_characters(st.session_state.data_chars, st.session_state.generator, True, initial_char_data = st.session_state.initial_char_data)
        
        elif st.session_state.prompting_approach == 'Custom':

            # print(f'INITIAL CHAR DATA BEFORE FUNCTION CALL: {st.session_state.initial_char_data}')
            st.session_state.story_generator.generate('characters', 4096, initial_character_data =  st.session_state.initial_char_data)
            st.session_state.data_chars['text'] = st.session_state.story_generator.characters
    
    elif st.session_state.char_gen == True and st.session_state.use_predefined_chars == False:
        if st.session_state.prompting_approach == 'DramaTron':
            fun_generate_characters(st.session_state.data_chars, st.session_state.generator)
        elif st.session_state.prompting_approach == 'Custom':
            st.session_state.story_generator.generate('characters', 4096, None)
            st.session_state.data_chars['text'] = st.session_state.story_generator.characters

    if st.session_state.char_regen == True \
    and st.session_state.initial_char_data is not None \
    and st.session_state.initial_char_data != '':
        
        if st.session_state.prompting_approach == 'DramaTron':
            fun_generate_characters(st.session_state.data_chars, st.session_state.generator, True, initial_char_data = st.session_state.initial_char_data)
        elif st.session_state.prompting_approach == 'Custom':
            st.session_state.story_generator.generate('characters', 4096, initial_character_data = st.session_state.initial_char_data)
            st.session_state.data_chars['text'] = st.session_state.story_generator.characters
    
    elif st.session_state.char_regen == True and st.session_state.use_predefined_chars == False:
        if st.session_state.prompting_approach == 'DramaTron':
            fun_generate_characters(st.session_state.data_chars, st.session_state.generator)
        
        elif st.session_state.prompting_approach == 'Custom':
            st.session_state.story_generator.generate('characters', 4096, None)
            
            st.session_state.data_chars['text'] = st.session_state.story_generator.characters 
 
    if st.session_state.data_chars is not None and st.session_state.data_chars['text'] != '':
        st.text_area(label = 'Characters', value = st.session_state.data_chars['text'])
        print(st.session_state.generator._characters)
        st.session_state.char_gen = False

#####################################################################################################
    st.write("<h4>Generate Scenes</h4>", unsafe_allow_html=True)
    if st.session_state.data_scenes['text'] == '':
        use_predefined_scenes = st.checkbox(label = 'Use predefined scenes?')
        st.session_state.scene_gen = st.button('Generate Scenes')
        
        if use_predefined_scenes == True:
            if st.session_state.initial_scenes_data_confirm == False:
                st.session_state.initial_scenes_data = st.text_input(label = 'Initial scenes description:')
                st.session_state.initial_scenes_data_confirm = st.button('Confirm the initial data')




    if st.session_state.data_scenes['text'] != '':
        st.session_state.scene_regen = st.button('Regenerate scenes.')

    if st.session_state.scene_gen == True\
        and st.session_state.initial_scenes_data != ''\
        and st.session_state.initial_scenes_data_confirm == True:
        if st.session_state.prompting_approach == 'DramaTron':
            raise NotImplementedError('DramaTron does not support initial data for scenes generation.')
            fun_generate_scenes(st.session_state.data_scenes, st.session_state.generator)   
        else:
            st.session_state.story_generator.generate('scenes', 4096, initial_scenes_data=st.session_state.initial_scenes_data)
            st.session_state.data_scenes['text'] = st.session_state.story_generator.scenes

    elif st.session_state.scene_gen == True and st.session_state.use_predefined_scenes == False:
        if st.session_state.prompting_approach == 'DramaTron':
            raise NotImplementedError('DramaTron does not support initial data for scenes generation.')
        else:
            st.session_state.story_generator.generate('scenes', 4096)
            st.session_state.data_scenes['text'] = st.session_state.story_generator.scenes


    
    if st.session_state.scene_regen == True and st.session_state.initial_scenes_data != '':
        if st.session_state.prompting_approach == 'DramaTron':
            Warning('DramaTron does not support initial data for scenes generation, the instructions will be ignored.')
            fun_generate_scenes(st.session_state.data_scenes, st.session_state.generator)   

        else:
            st.session_state.story_generator.generate('scenes', 4096, initial_scenes_data=st.session_state.initial_scenes_data)
            st.session_state.data_scenes['text'] = st.session_state.story_generator.scenes
    elif st.session_state.scene_regen == True and st.session_state.use_predefined_scenes == False:
        if st.session_state.prompting_approach == 'DramaTron':
            Warning('DramaTron does not support initial data for scenes generation, the instructions will be ignored.')
            fun_generate_scenes(st.session_state.data_scenes, st.session_state.generator)  
        else:
            st.session_state.story_generator.generate('scenes', 4096)
            st.session_state.data_scenes['text'] = st.session_state.story_generator.scenes


    if st.session_state.data_scenes is not None and st.session_state.data_scenes['text'] != '':
        if st.session_state.prompting_approach == 'DramaTron':
            st.text_area(label = 'Scenes', value = st.session_state.generator._scenes)
        else:
            st.text_area(label = 'Scenes', value = st.session_state.story_generator.scenes[0])
        st.session_state.num_scenes = st.session_state.generator.num_scenes()

        if len(st.session_state.list_of_data_dialogs) != st.session_state.num_scenes:
            for i in range(st.session_state.num_scenes):
                data_dialogs = {
                    "lock": False,
                    "text_area": None,
                    "raw_text" : None,
                    "seed": st.session_state.generator.seed - 1,
                    "history": [GenerationHistory() for _ in range(st.session_state.num_scenes)],
                    "scene": i+1
                }    
                st.session_state.list_of_data_dialogs.append(data_dialogs)
        st.session_state.scene_gen = False


#####################################################################################################
    st.write("<h4>Generate Places</h4>", unsafe_allow_html=True)
    if st.session_state.data_places['descriptions'] == st.session_state.place_descriptions:
        use_predefined_places = st.checkbox(label = 'Use predefined places?')
        st.session_state.initial_places_images = st.checkbox(label = 'Use initial places data as images')
        print(f'initial_places_images: {st.session_state.initial_places_images}')
        if st.session_state.initial_places_images == True:
            use_predefined_places = True
        print(f'Flags: {st.session_state.initial_places_images} - {use_predefined_places}')
        if use_predefined_places == True:
            if st.session_state.initial_places_images == False:
                if st.session_state.initial_places_data_confirm == False:
                    st.session_state.initial_places_data = st.text_input(label = 'initial places descriptions:')
                    st.session_state.initial_places_data_confirm = st.button('Confirm the description')
            elif st.session_state.initial_places_images == True:
                temp_img_path = 'temp.jpg'
                if st.session_state.initial_places_data_confirm == False:
                    st.session_state.files = st.file_uploader('Upload the images of places', accept_multiple_files=True)
                    st.session_state.upload_places = st.button(label = 'Upload places images')

                    if st.session_state.upload_places == True:
                        st.session_state.initial_places_data = ''
                        list_of_output_results = []
                        for ind, item in enumerate(st.session_state.files):
                            byte_io = io.BytesIO(item.read())
                            image = Image.open(byte_io)
                            image.save(temp_img_path)
                            conv_result = st.session_state.img2text.convert(temp_img_path)
                            list_of_output_results.append(f'Place {ind} - Descriptin: {conv_result}')
                        st.session_state.initial_places_data = '\n\n'.join(list_of_output_results)
                        print(st.session_state.initial_places_data)
                        st.session_state.initial_places_data_confirm = st.button('Confirm the description')

        st.session_state.places_gen = st.button('Generate Place')

    
    elif st.session_state.data_places['descriptions'] != st.session_state.place_descriptions:
        st.session_state.places_regen = st.button('Regenerate Places')

    if st.session_state.places_gen == True\
        and st.session_state.initial_places_data !='':
        if st.session_state.prompting_approach == 'DramaTron':
            fun_generate_places(st.session_state.data_places, st.session_state.generator)
            Warning('DramaTron does not support the places as initial data')
        else:
            st.session_state.story_generator.generate('places', 4096, initial_places_data=st.session_state.initial_places_data)
            st.session_state.data_places['descriptions'] = st.session_state.story_generator.places
    elif st.session_state.places_gen == True and st.session_state.use_predefined_places == False:
        if st.session_state.prompting_approach == 'DramaTron':
            fun_generate_places(st.session_state.data_places, st.session_state.generator)
        else:
            st.session_state.story_generator.generate('places', 4096)
            st.session_state.data_places['descriptions'] = st.session_state.story_generator.places


    if st.session_state.places_regen == True\
       and st.session_state.initial_places_data !='':
        if st.session_state.prompting_approach == 'DramaTron':
            fun_generate_places(st.session_state.data_places, st.session_state.generator)
            Warning('DramaTron does not support the places as initial data')
        else:
            st.session_state.story_generator.generate('places', 4096, initial_places_data = st.session_state.initial_places_data)
            st.session_state.data_places['descriptions'] = st.session_state.story_generator.places
    
    elif st.session_state.places_regen == True and st.session_state.use_predefined_places == False:
        if st.session_state.prompting_approach == 'DramaTron':
            fun_generate_places(st.session_state.data_places, st.session_state.generator)
        else:
            st.session_state.story_generator.generate('places', 4096)
            st.session_state.data_places['descriptions'] = st.session_state.story_generator.places

    if st.session_state.data_places is not None and st.session_state.data_places['descriptions'] != st.session_state.place_descriptions:
        st.text_area(label = 'Places', value = st.session_state.data_places['descriptions'])
        st.session_state.places_gen = False
        

#####################################################################################################
    if st.session_state.data_places is not None and st.session_state.data_places['descriptions'] != st.session_state.place_descriptions:
        st.write("<h4>Generate Dialogs</h4>", unsafe_allow_html=True)

        is_empty_counter = check_empty_dialog_list(st.session_state.list_of_data_dialogs)
        print(f'EMPTY COUNTER _1: {is_empty_counter}')
        if is_empty_counter == len(st.session_state.list_of_data_dialogs):
            st.session_state.gen_dialogs = st.button('Generate Dialogs')
        if is_empty_counter != len(st.session_state.list_of_data_dialogs):
            st.session_state.regen_dialogs = st.button('Regenerate Dialogs')
        
        if st.session_state.gen_dialogs == True:
            if st.session_state.prompting_approach == 'DramaTron':
                for item in st.session_state.list_of_data_dialogs:
                    st.session_state.list_of_data_dialogs_filled.append(fun_generate_dialog(item, st.session_state.generator))
            else:
                st.session_state.story_generator.generate('dialog', 4096)
                st.session_state.list_of_data_dialogs_filled = st.session_state.story_generator.list_of_dialogs
        
        if st.session_state.regen_dialogs == True:
            st.session_state.list_of_data_dialogs_filled = []
            if st.session_state.prompting_approach == 'DramaTron':
                for item in st.session_state.list_of_data_dialogs:
                    st.session_state.list_of_data_dialogs_filled.append(fun_generate_dialog(item, st.session_state.generator))
            else:
                st.session_state.story_generator.generate('dialog', 4096)
                st.session_state.list_of_data_dialogs_filled = st.session_state.story_generator.list_of_dialogs

        # print(f'LIST OF DATA DIALOGS: {st.session_state.list_of_data_dialogs_filled}')
        is_empty_counter_ = check_empty_dialog_list(st.session_state.list_of_data_dialogs_filled)
        print(f'EMPTY COUNTER: {is_empty_counter_}')
        if is_empty_counter_ == 0:
            st.session_state.gen_dialogs = False
           
            if st.session_state.prompting_approach == 'DramaTron':
                dialog_index = st.slider('Pick the dialog to show.', min_value=1, max_value=len(st.session_state.list_of_data_dialogs), step=1)
                st.text_area(label = 'Dialog', value = st.session_state.list_of_data_dialogs[dialog_index-1]['raw_text'])
            else:
                dialog_index = st.slider('Pick the dialog to show.', min_value=1, max_value=len(st.session_state.list_of_data_dialogs_filled), step=1)
                st.text_area(label = 'Dialog', value = st.session_state.list_of_data_dialogs_filled[dialog_index-1])
            parse = st.button('Parse the dialog')
            
            if parse == True:
                prompt_to_send = create_prompt_simple(st.session_state.generator._dialogs[dialog_index-1])
                output = send_request(prompt_to_send, model_name = 'gpt-4o-mini')
                print(output)
                list_of_names = list(st.session_state.generator._characters[0].keys())
                list_of_names = [item.upper()+'  ' for item in list_of_names]
                output_dict = parse_output(list_of_names, output)
                st.text_area(label = "Parsed dialog.", value = output_dict)
                with open('output_dict.pkl', 'wb') as file:
                    pickle.dump(output_dict, file)

##################################################################################################
            st.write("<h4>Render the story</h4>", unsafe_allow_html=True)
            if st.session_state.prompting_approach == 'DramaTron':
                story = st.session_state.generator.get_story()
                st.session_state.story = render_story(story)
                st.text_area(label = 'Story', value = st.session_state.story)
                with open('rendered_story.txt', 'w') as file:
                    file.write(st.session_state.story)
            else:
                st.session_state.story = st.session_state.story_generator.get_story()
                st.text_area(label = 'Story', value = st.session_state.story)
                with open('rendered_story.txt', 'w') as file:
                    file.write(st.session_state.story)

            if  st.session_state.image_generator is not None:
                dict_of_characters_desc = st.session_state.story_generator.get_descriptions('characters')
                dict_of_places_desc = st.session_state.story_generator.get_descriptions('places')
                

                dict_of_output_characters = st.session_state.image_generator.generate_images(dict_of_characters_desc)
                dict_of_output_places = st.session_state.image_generator.generate_images(dict_of_places_desc)
                gen_list_of_dicts, gen_vis_list, gen_text_list, gen_char_list = st.session_state.story_generator.get_storyboards()
                
                
                os.makedirs('generated_images/characters', exist_ok = True)
                os.makedirs('generated_images/places', exist_ok = True)
                os.makedirs('generated_images/storyboards', exist_ok = True)

                for dict_ind in range(len(gen_list_of_dicts)):
                    dict_to_process = gen_list_of_dicts[dict_ind]
                    dict_of_generated_images = st.session_state.img_gen.generate_images(dict_to_process, style_of_generation = 'ULTRA REALISTIC.')
                    out_path = os.path.join(base_dir, str(dict_ind))
                    os.makedirs(out_path, exist_ok = True)
                    save_images(out_path, dict_of_generated_images)

                save_images('generated_images/characters', dict_of_output_characters)
                save_images('generated_images/places', dict_of_output_places)

                st.write('Images were generated. To view them visit the Show And Edit images page')






        



