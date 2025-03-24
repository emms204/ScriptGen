import streamlit as st
from src.ImageProcessing.ImageProcessor import FaceProcessor
import os
import io
from PIL import Image


output_folder = '/workspace/face_anim_project/ref_pictures'
output_folder = 'ref_pictures'
temp_file_path = '/workspace/face_anim_project/temp.jpg'
temp_file_path = 'temp.jpg'


if 'file_name' not in st.session_state.keys():
    st.session_state.file_name = None
    st.session_state.confirm_upload = False
    st.session_state.file = False
    st.session_state.processed_image = None
    st.session_state.img_processor = FaceProcessor()
    st.session_state.img_for_preview_path = None
    st.session_state.switcher = True
    
    st.session_state.loaded_img = None
    st.session_state.show_preview_1 = None
    st.session_state.close_preview_1 = None
    


st.title('Upload the reference images for animation.')

list_of_available_files = os.listdir(output_folder)

st.session_state.img_for_preview_path = st.selectbox('List of available reference files', list_of_available_files)
if st.session_state.img_for_preview_path not in [None, '']:

    if st.session_state.switcher == True:
        st.session_state.show_preview_1 = st.button('Show Preview')
    
    if st.session_state.show_preview_1 == True:
        st.session_state.switcher = False
        st.session_state.close_preview_1 = st.button("Close preview")
    
    if st.session_state.close_preview_1 == True: #WORKS FINE WITHOUT THIS BLOCK
        
        st.session_state.show_preview_1 = False
        st.session_state.switcher = True
        

    if st.session_state.show_preview_1 == True:
        st.image(os.path.join(output_folder, st.session_state.img_for_preview_path))



st.session_state.file_name = st.text_input(label = 'Please enter how the file should be named.')

st.session_state.file = st.file_uploader('Upload the reference audio:', type=['.jpg', '.png'])

if st.session_state.file is not None and st.session_state.file_name is not None:
    byte_io = io.BytesIO(st.session_state.file.read())
    image = Image.open(byte_io)
    image.save(temp_file_path)
    
    message, bboxes = st.session_state.img_processor.detect_faces(temp_file_path)

    if message == 'OK!':
        st.write(message)
        st.session_state.img_processor.crop_faces(temp_file_path,
                                                  bboxes,
                                                  os.path.join(output_folder, st.session_state.file_name))
        
        st.image(os.path.join(output_folder, st.session_state.file_name))

    else:
        st.write(message)
        st.write('Try another picture')

    