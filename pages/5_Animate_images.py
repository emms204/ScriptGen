import streamlit as st
import pickle
from src.TextGen.utils import *
import pandas as pd
import os


def convert_ref_dict(ref_dict):
    conv_dict = {}
    for key in list(ref_dict.keys()):
        if key == 'Character name':
            for nest_key in list(ref_dict[key].keys()):
                conv_dict[ref_dict[key][nest_key]] = ref_dict['Reference image path'][nest_key]
    
    return conv_dict



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

ref_img_folder = 'ref_pictures'

st.title('Animation generation')

# try:
if st.session_state.loaded_dictionary is None:
    with open('output_dict.pkl', 'rb') as file:
        st.session_state.loaded_dictionary = pickle.load(file)
    st.text('Dictionary was loaded.')
    st.text_area(label = 'Parsed dialog.', value = st.session_state.loaded_dictionary)
else:
    st.text('Dictionary was loaded.')
    st.text_area(label = 'Parsed dialog.', value = st.session_state.loaded_dictionary)
# except Exception as e:
#     st.write('The generated dictionary cannot be located.')


test_dict = {"SIR CEDRIC  ": {1 : 'some_1.wav', 2 : 'some_2.wav', 3 : 'some_3.wav'},
            'LADY ELARA  ': {1 : 'some_1_elara.wav', 2 : 'some_2_elara.wav', 3 : 'some_3_elara.wav'}}


list_of_available_files = os.listdir(ref_img_folder)
st.selectbox('List of available reference files', list_of_available_files)

if st.session_state.confirm_ref_dict_creation == False:
    st.session_state.reference_dictionary['Character name'] = []
    for key in list(st.session_state.loaded_dictionary.keys()):
        st.session_state.reference_dictionary['Character name'].append(key)  
    st.session_state.reference_dictionary['Reference image path'] = ['' for i in range(len(st.session_state.loaded_dictionary.keys()))]

    st.session_state.reference_df = pd.DataFrame(data = st.session_state.reference_dictionary, index = list(range(len(st.session_state.loaded_dictionary.keys()))))

if st.session_state.confirm_ref_dict_creation == False:
    st.session_state.reference_df = st.data_editor(st.session_state.reference_df)
    st.session_state.confirm_ref_dict_creation = st.button(label = 'Confirm')

    if st.session_state.confirm_ref_dict_creation == True:
        st.session_state.reference_dictionary = st.session_state.reference_df.to_dict()
        st.session_state.reference_dictionary = convert_ref_dict(st.session_state.reference_dictionary)
        map_dict = st.session_state.reference_dictionary
        st.text(map_dict)

        output_dict = map_the_voices_and_pictures(map_dict, test_dict)
        
        st.text(output_dict)
        st.write('Animation generation coming soon!')