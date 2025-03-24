import streamlit as st
import soundfile as sf
import os
import io

output_folder = '/workspace/face_anim_project/ref_audio'
output_folder = 'ref_audio'

if 'choice_of_model' not in st.session_state.keys():
    st.session_state.choice_of_model = None
    st.session_state.file_name = None
    st.session_state.confirm_upload = False
    st.session_state.file = False
    st.session_state.reference_transcript = None

st.title('Upload the reference audios for CoquiTTS and Mars5TTS')


st.session_state.choice_of_model = st.selectbox('Choose the model type:', ['CoquiTTS', 'Mars5TTS'], index = 0)
list_of_available_files = os.listdir(output_folder)
st.selectbox('List of available reference files', list_of_available_files)



if st.session_state.choice_of_model == 'CoquiTTS':
    st.write('For this type of model please provide a reference audio of any length in .wav format. The model tends to perform better on the longer examples.')
    st.session_state.file_name = st.text_input(label = 'Please enter how the file should be named.')
    st.session_state.file = st.file_uploader('Upload the reference audio:', type='.wav')

    if st.session_state.file is not None:
        byte_io = io.BytesIO(st.session_state.file.read())
        audio_data, sampling_rate = sf.read(byte_io)
        

        sf.write(os.path.join(output_folder, st.session_state.choice_of_model + '_' + st.session_state.file_name), audio_data, sampling_rate)
        place_holder = st.empty()
        place_holder.markdown('File was saved.')

if st.session_state.choice_of_model == 'Mars5TTS':
    st.write('This type of model requires the reference audio and correspondive reference transcript. Length of the audio should be from 3 to 10 seconds.')
    st.session_state.file_name = st.text_input(label = 'Please enter how the file should be named.')
    st.session_state.reference_transcript = st.text_input(label = 'Please enter the reference transcript.')
    txt_file_name = st.session_state.file_name.split('.')[0]+'.txt'
    st.session_state.file = st.file_uploader('Upload the reference audio:', type='.wav')

    if st.session_state.file is not None and st.session_state.reference_transcript != '' and st.session_state.reference_transcript is not None:
        byte_io = io.BytesIO(st.session_state.file.read())
        audio_data, sampling_rate = sf.read(byte_io)
        

        sf.write(os.path.join(output_folder, st.session_state.choice_of_model + '_' + st.session_state.file_name), audio_data, sampling_rate)
        with open(os.path.join(output_folder, st.session_state.choice_of_model + '_' +txt_file_name), 'w') as file:
            file.write(st.session_state.reference_transcript)
        place_holder = st.empty()
        place_holder.markdown('File was saved.')


