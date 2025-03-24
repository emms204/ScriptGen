import streamlit as st
import os


st.title('Images view and editing are coming soon.')


if 'output_choice' not in st.session_state.keys():
    st.session_state.output_choice = None
    st.session_state.output_choice_confirm = False
    st.session_state.image_number = None
    st.session_state.folder_to_check = None




st.session_state.output_choice = st.selectbox('What images do you want to check?', ['Places', 'Characters', 'StoryBoards'])
if st.session_state.output_choice_confirm == False:
    st.session_state.output_choice_confirm = st.button('Confirm your choice.')

if st.session_state.output_choice_confirm == True:
    if st.session_state.output_choice == 'Places':
        path_to_images = 'generated_images/places'
    elif st.session_state.output_choice == 'Characters':
        path_to_images = 'generated_images/characters'
    elif st.session_state.output_choice == 'StoryBoards':
        temp_path = 'generated_images/storyboards'
        st.session_state.folder_to_check = st.selectbox('Pick the folder for view', options=list(os.listdir(temp_path)))
        path_to_images = os.path.join(temp_path, st.session_state.folder_to_check)
    
    list_of_files = list(os.listdir(path_to_images))
    list_of_images = [item for  item in list_of_files if '.jpg' in item]
    list_of_descs = [item for item in list_of_files if '.txt' in item] 
    

    st.session_state.image_name = st.selectbox(label = 'Choose the image to show.', options = list_of_images)
    if st.session_state.image_name is not None:
        st.image(os.path.join(path_to_images, st.session_state.image_name))
        txt_file = st.session_state.image_name.replace('jpg', 'txt')
        with open(os.path.join(), 'r') as file:
            text_to_print = file.read()
        place_holder_text = st.empty()
        place_holder_text.markdown(f'<p>{text_to_print}</p>')
    # st.session_state.image_number = st.slider(label = 'Pick the image to show', min_value=1, max_value = len(list_of_images), step = 1)
    # img_to_show = list_of_images[st.session_state.image_number-1]
    # st.image(os.path.join(path_to_images, img_to_show))
    