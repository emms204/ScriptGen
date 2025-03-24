import streamlit as st
from TextGen.dramatron.DramaTronCore import *
from src.TextGen.HyperParameters import *
from src.TextGen.utils import *
from src.TextGen.TextGen import *



if 'model_name' not in st.session_state.keys():
    st.session_state.model_name = None
    st.session_state.model_name_selected = False
    st.session_state.client = None
    st.session_state.prompt_entered = False
    st.session_state.prompt = None

st.title('Process generated script')

#load script

st.session_state.model_name = st.selectbox('Choose the model type:', ['gpt', 'sonnet'], index = 0,)
if st.session_state.model_name_selected == False:
    st.session_state.model_name_selected = st.button('Confirm your choise')



if st.session_state.model_name_selected == True:
    st.session_state.client  = DramatronHandler(
        model_type=st.session_state.model_name,
        seed=DEFAULT_SEED,
        sample_length=config['sample_length'],
        max_retries=config['max_retries'],
        config_sampling=config['sampling'],
        timeout=TIMEOUT)
    


    st.write("<h5>Read and display the generated script</h5>", unsafe_allow_html=True)
    with open('rendered_story.txt', 'r') as f:
        output_script_text = f.read()

    st.text_area(label = 'Generated script', value = output_script_text)

    st.session_state.prompt =  st.text_input('Input the prompt', value = 'Can you define any weak spots of the script?')
    st.session_state.prompt_entered =  st.button(label = 'Confirm')

    if st.session_state.prompt_entered == True:
        processed_script =  st.session_state.client.process_script(output_script_text,True)

        output_of_the_model = st.session_state.client.give_recommendations(processed_script['output_text'], prompt = st.session_state.prompt)

        st.text_area(label = 'Recommendations of the LLM:', value=output_of_the_model)

print(st.session_state.model_name_selected)