import streamlit as st
from src.SoundGen.SoundGen import ParlerTTS, CoquiTTS, Mars5TTS
import pickle
import pandas as pd
import os
import torchaudio
import glob
import gc
import torch

# Clear PyTorch cache
torch.cuda.empty_cache()

# Force garbage collection
gc.collect()

def convert_parl_ref_dict(ref_dict):
    conv_dict = {}
    for key in list(ref_dict.keys()):
        if key == 'Character name':
            for nest_key in list(ref_dict[key].keys()):
                conv_dict[ref_dict[key][nest_key]] = ref_dict['Reference description'][nest_key]

    return conv_dict

def convert_coqui_ref_dict(ref_dict):
    conv_dict = {}
    for key in list(ref_dict.keys()):
        if key == 'Character name':
            for nest_key in list(ref_dict[key].keys()):
                conv_dict[ref_dict[key][nest_key]] = ref_dict['Path to reference audio'][nest_key]
    return conv_dict

def convert_mar5tts_ref_dict(ref_dict):
    conv_dict = {}
    for key in list(ref_dict.keys()):
        if key == 'Character name':
            for nest_key in list(ref_dict[key].keys()):
                conv_dict[ref_dict[key][nest_key]] = [ref_dict['Path to reference audio'][nest_key], ref_dict['Reference transcript'][nest_key]]
    return conv_dict

def display_audios(list_of_pathes, list_of_lines):

    for audio_file, line in zip(list_of_pathes, list_of_lines):
        format = "audio/wav"
        
        write_object = st.write(line)
        audio_object = st.audio(audio_file, format=format)

def get_the_lines(dialogue_dict):
    combined_lines = []

    for character, lines in dialogue_dict.items():
        for index, line in lines.items():
            combined_lines.append((index, f"{character.strip()}: {line}"))

    # Sort the combined list by the index
    sorted_lines = sorted(combined_lines)

    # Extract and print the sorted dialogue lines
    ordered_dialogue = [line for _, line in sorted_lines]

    return ordered_dialogue

def delete_files(path_to_folder):
    files = glob.glob(os.path.join(path_to_folder, '*'))

    for file in files:
        os.remove(file)




if 'model_name' not in st.session_state.keys():
    st.session_state.model_name = None
    st.session_state.model_name_selected = False
    st.session_state.client = None


if 'loaded_dictionary' not in st.session_state.keys():
    st.session_state.prompt_entered = False
    st.session_state.prompt = None
    st.session_state.loaded_dictionary = None
    st.session_state.reference_dictionary = dict()
    

if 'reference_df' not in st.session_state.keys():
    st.session_state.reference_df = None
    st.session_state.reference_dictionary_created = False

if 'confirm_ref_dict_creation' not in st.session_state.keys():
    st.session_state.confirm_ref_dict_creation = False
    st.session_state.file_preview_choice = False
    st.session_state.file_preview = None

if 'sounds_generated' not in st.session_state.keys():
    st.session_state.sounds_generated = False
    st.session_state.sound_generator = None


st.title('Voice generation')

try:

    if st.session_state.loaded_dictionary is None:
        with open('output_dict.pkl', 'rb') as file:
            st.session_state.loaded_dictionary = pickle.load(file)
    else:
        st.text('Dictionary was loaded.')
        st.text_area(label = 'Parsed dialog.', value = st.session_state.loaded_dictionary)
except Exception as e:
    st.write('The generated dictionary cannot be located.')

ref_audio_folder = 'ref_audio'





if st.session_state.model_name_selected == False:
    st.session_state.model_name = st.selectbox('Choose the model type:', ['ParlerTTS', 'CoquiTTS', 'Mars5TTS'], index = 0)
    st.session_state.model_name_selected = st.button('Confirm your choise')

list_of_available_files = os.listdir(ref_audio_folder)
st.selectbox('List of available reference files', list_of_available_files)

if st.session_state.model_name == 'ParlerTTS' and st.session_state.model_name_selected == True:
    if st.session_state.confirm_ref_dict_creation == False:
        if st.session_state.sound_generator is None:
            st.session_state.sound_generator = ParlerTTS()

        st.session_state.reference_dictionary['Character name'] = []
        for key in list(st.session_state.loaded_dictionary.keys()):
            st.session_state.reference_dictionary['Character name'].append(key)  
        st.session_state.reference_dictionary['Reference description'] = ['' for i in range(len(st.session_state.loaded_dictionary.keys()))]


    st.session_state.reference_df = pd.DataFrame(data = st.session_state.reference_dictionary, index = list(range(len(st.session_state.loaded_dictionary.keys()))))
    st.write("<h5>Create a reference dictionary for ParlerTTS</h5>", unsafe_allow_html=True)
    st.write("""Possible options to use:
                <ul>
             <li>Gary's voice is monotone yet slightly fast in delivery, with a very close recording that has no background noise.</li>
             <li>Jen's voice is monotone yet slightly fast in delivery, with a very close recording that has no background noise.</li>
             </ul>""", unsafe_allow_html=True)
    
    if st.session_state.confirm_ref_dict_creation == False:
        st.session_state.reference_df = st.data_editor(st.session_state.reference_df)
        st.session_state.confirm_ref_dict_creation = st.button(label = 'Confirm')

    if st.session_state.confirm_ref_dict_creation == True:
        
        st.session_state.reference_dictionary = st.session_state.reference_df.to_dict()
        st.text(st.session_state.reference_dictionary)

        st.write('Start of the sound genertion!')
        if st.session_state.sounds_generated == False:
            conv_dict = convert_parl_ref_dict(st.session_state.reference_dictionary)
            output_folder_path = 'generated_audios'
            delete_files(output_folder_path)
            st.session_state.sound_generator.general_inference(st.session_state.loaded_dictionary, conv_dict, output_folder = '/workspace/face_anim_project/generated_audios/')
            
            st.session_state.sounds_generated = True
            

        
    
    

elif st.session_state.model_name == 'CoquiTTS' and st.session_state.model_name_selected == True:
    if st.session_state.confirm_ref_dict_creation == False:

        if st.session_state.sound_generator is None:
            st.session_state.sound_generator = CoquiTTS()
            
        st.session_state.reference_dictionary['Character name'] = []
        for key in list(st.session_state.loaded_dictionary.keys()):
            st.session_state.reference_dictionary['Character name'].append(key)  
        st.session_state.reference_dictionary['Path to reference audio'] = ['' for i in range(len(st.session_state.loaded_dictionary.keys()))]

    st.session_state.reference_df = pd.DataFrame(data = st.session_state.reference_dictionary, index = list(range(len(st.session_state.loaded_dictionary.keys()))))
    st.write("<h5>Create a reference dictionary for CoquiTTS</h5>", unsafe_allow_html=True)
    st.write("""Possible options to use:
                <ul>
             <li>/workspace/face_anim_project/ref_audio/male_ref_2_min.wav</li>
             <li>/workspace/face_anim_project/ref_audio/female_ref_2_min.wav</li>
             </ul>""", unsafe_allow_html=True)
    
    
    if st.session_state.confirm_ref_dict_creation == False:
        st.session_state.reference_df = st.data_editor(st.session_state.reference_df)
        st.session_state.confirm_ref_dict_creation = st.button(label = 'Confirm')

    if st.session_state.confirm_ref_dict_creation == True:
        
        st.session_state.reference_dictionary = st.session_state.reference_df.to_dict()
        st.text(st.session_state.reference_dictionary)

        st.write('Start of the sound genertion!')
        if st.session_state.sounds_generated == False:
            output_folder_path = '/workspace/face_anim_project/generated_audios/'
            delete_files(output_folder_path)
            conv_dict = convert_coqui_ref_dict(st.session_state.reference_dictionary)
            st.session_state.sound_generator.inference(st.session_state.loaded_dictionary, output_folder=output_folder_path, dict_with_mapped_chars = conv_dict)
            
            st.session_state.sounds_generated = True



elif st.session_state.model_name == 'Mars5TTS' and st.session_state.model_name_selected == True:
    if st.session_state.confirm_ref_dict_creation == False:
        if st.session_state.sound_generator is None:
            st.session_state.sound_generator = Mars5TTS()
            
        st.session_state.reference_dictionary['Character name'] = []
        for key in list(st.session_state.loaded_dictionary.keys()):
            st.session_state.reference_dictionary['Character name'].append(key)  
        st.session_state.reference_dictionary['Path to reference audio'] = ['' for i in range(len(st.session_state.loaded_dictionary.keys()))]
        st.session_state.reference_dictionary['Reference transcript'] = ['' for i in range(len(st.session_state.loaded_dictionary.keys()))]

    st.session_state.reference_df = pd.DataFrame(data = st.session_state.reference_dictionary, index = list(range(len(st.session_state.loaded_dictionary.keys()))))

    st.write("<h5>Create a reference dictionary for Mars5TTS</h5>", unsafe_allow_html=True)
    st.write("""Possible options to use:
                <ul>
             <li>Path to reference audio: /workspace/face_anim_project/ref_audio/male_ref_2_16.wav ; Reference transcript: "Two thousand twelve and quite frankly the fidelity was pretty crappy. You're lucky to sound this good. Okay.";</li>
             <li>Path to reference audio: /workspace/face_anim_project/ref_audio/female_ref_2_16.wav; Reference transcript: I am curious. I am grateful. I am strong. I am grounded. I am free.;</li>
             </ul>""", unsafe_allow_html=True)
    
    list_of_files_2 = list(os.listdir(ref_audio_folder))
    list_of_txt = [item for item in list_of_files_2 if  'txt' in item ]
    
    if st.session_state.confirm_ref_dict_creation == False:
        st.session_state.file_preview_choice = st.selectbox(label = 'Reference transcript preview', options = list_of_txt)
        file_path_to_open  = os.path.join(ref_audio_folder, st.session_state.file_preview_choice)
        ref_transcript = None
        with open(file_path_to_open, 'r') as file:
            ref_transcript = file.read()

        if ref_transcript is not None:
            st.text(f'Reference transcript: {ref_transcript}')
        
        st.session_state.reference_df = st.data_editor(st.session_state.reference_df)
        st.session_state.confirm_ref_dict_creation = st.button(label = 'Confirm')

    if st.session_state.confirm_ref_dict_creation == True:

        st.session_state.reference_dictionary = st.session_state.reference_df.to_dict()
        st.text(st.session_state.reference_dictionary)

        st.write('Start of the sound generation!')
        if st.session_state.sounds_generated == False:
            output_folder_path = '/workspace/face_anim_project/generated_audios/'
            delete_files(output_folder_path)
            conv_dict = convert_mar5tts_ref_dict(st.session_state.reference_dictionary)

            st.session_state.sound_generator.inference(st.session_state.loaded_dictionary, conv_dict, output_folder = output_folder_path)
            st.session_state.sounds_generated = True





if st.session_state.sounds_generated == True:

    path_to_the_output_folder = 'generated_audios'
    list_of_files = sorted(list(os.listdir('generated_audios')), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    list_of_files = [os.path.join(path_to_the_output_folder, file) for file in list_of_files]
    list_of_lines = get_the_lines(st.session_state.loaded_dictionary)
    print(list_of_files)

    display_audios(list_of_files, list_of_lines)
    del st.session_state.sound_generator