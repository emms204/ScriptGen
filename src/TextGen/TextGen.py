
import os
from dotenv import load_dotenv
import google.generativeai as genai
import os
import anthropic
from ollama import chat
from ollama import ChatResponse
import subprocess
from transformers import pipeline
import torch
import re
from langchain.docstore.document import Document
from HyperParameters import *
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.TextGen.utils import *
from TextGen.dramatron.DramaTronCore import *
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_openai import ChatOpenAI
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from TextGen.examples import example_of_scenes
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from src.TextGen.prompts import place_prompt_template, dilog_prompt_template, dialog_prompt_template_sequence, title_format_template,\
    characters_prompt_template, characters_prompt_template_with_initial_data, scenes_prompt_template, place_prompt_template_with_II,\
    storyboards_prompt_template
from src.TextGen.prompts import scenes_prompt_template_with_II
from groq import Groq
from src.TextGen.prompts import toxicity_template, regenerate_template

load_dotenv()

OPEN_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SONNET_API_KEY = os.getenv('SONNET_API_KEY')
try:
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
except:
    DEEPSEEK_API_KEY = None



class DramatronHandler(LanguageAPI):
    def __init__(self, model_type, sample_length, seed, max_retries, config_sampling, timeout, path_to_models_weights = None, config = None, ollama_model_name = None) -> None:
        self.model_type = model_type
        self.ollama_model_name = ollama_model_name
        self.frequency_penalty = 0.2
        self.presence_penalty = 0.2
        self.system_prompt = 'You are a creative writing assistant for a team of writers. Your goal is to expand on the input text prompt and to generate the continuation of that text without any comments. Be as creative as possible, write rich detailed descriptions and use precise language. Add new original ideas. Finish generation with **END**.'
        
        if model_type == 'gpt':
            self.model_name = 'gpt-4o'
            self.system_prompt = 'You are a helpful playwright assistant.'
            self.client = OpenAI(api_key= OPEN_API_KEY)

            
          
        elif model_type == 'gemini':
            genai.configure(api_key=GOOGLE_API_KEY)
            self.client = genai.GenerativeModel('gemini-1.5-pro', system_instruction = self.system_prompt)
            self.model_name = 'gemini-1.5-pro'
              
           
        elif model_type == 'sonnet':
            self.client = anthropic.Anthropic(api_key=SONNET_API_KEY,)
            self.model_name = "claude-3-5-sonnet-20240620"


        elif model_type == 'llama-3.1-70b-quant':
            self.model_name = "meta-llama/Meta-Llama-3.1-70B"
            self.tokenizer = AutoTokenizer.from_pretrained('NousResearch/Hermes-3-Llama-3.1-70B', trust_remote_code=True)
            model = LlamaForCausalLM.from_pretrained(
                "NousResearch/Hermes-3-Llama-3.1-70B",
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=False,
                load_in_4bit=True,
                use_flash_attention_2=True
            )
            self.client = model

        elif model_type == 'llama-3.1-70b':
            self.model_name = 'llama-3.1-70b-versatile'
            self.client = Groq()

        elif model_type == 'mixtral-8x7b':
            self.model_name = 'mixtral-8x7b-32768'
            self.client = Groq()

        elif model_type == 'deep-seek-api' or model_type == 'deep-seek-api-reasoning':
            self.model_name = 'DeepSeek'
            self.system_prompt = 'You are a helpful playwright assistant.'
            self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


        elif model_type == 'deep-seek-local' or 'qwen':
            self.system_prompt = 'You are a helpful playwright assistant.'
            if ollama_model_name is not None:
                result = subprocess.run(f'ollama pull {ollama_model_name}', shell = True)
                print(f"Output of ollama PULL command: {result.stdout}")
            else:
                raise AssertionError('ollama_model_name argument was not provided.')
            
            result = subprocess.run(f'ollama run {ollama_model_name}')
            print(f'OUTPUT OF Ollama run command: {result.stdout}')

            

        #CAN BE DONE BETTER
        elif model_type == 'openai-o1':
            self.system_prompt = 'You are a helpful playwright assistant.'
            self.model_name = 'o1'
            self.client = OpenAI(api_key= OPEN_API_KEY)


        elif model_type == 'openai-o1-mini':
            self.system_prompt = 'You are a helpful playwright assistant.'
            self.model_name = 'o1-mini'
            self.client = OpenAI(api_key= OPEN_API_KEY)

        elif model_type == 'openai-o3-mini':
            self.system_prompt = 'You are a helpful playwright assistant.'
            self.model_name = 'o3-mini'
            self.client = OpenAI(api_key= OPEN_API_KEY)


        
        elif model_type == 'llama-3.1-405b':
            self.model_name = 'meta-llama/Meta-Llama-3.1-405B'
            self.client = pipeline("text-generation", model = self.model_name, model_kwargs={"torch_dtype":torch.bfloat16}, device_map = "auto")

        elif model_type == 'mistral_large':
            #A100 PCI values 
            self.model_name = 'Mistral-Large-Instruct-2407.Q5_K_M'
            n_gpu_layers = 80 
            n_batch = 128  
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            self.client = LlamaCpp(
                model_path=path_to_models_weights,
                n_gpu_layers=n_gpu_layers,
                n_batch=n_batch,
                # temperature = config['sampling']['temp'],
                # max_tokens=config['max_paragraph_length'],
                # top_p=config['sampling']['prob'],
                callback_manager=callback_manager,
                n_ctx = 16384,
                # verbose=True,  
            )
        
        elif model_type == 'LongWriter-glm4-9b':
            self.model_name = 'LongWriter-glm4-9b'
            self.tokenizer = AutoTokenizer.from_pretrained("THUDM/LongWriter-glm4-9b", trust_remote_code=True)
            self.client = AutoModelForCausalLM.from_pretrained("THUDM/LongWriter-glm4-9b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
            self.client = self.client.eval()

        elif model_type == 'LongWriter-llama3.1-8b':
            self.model_name = 'LongWriter-llama3.1-8b'
            self.tokenizer = AutoTokenizer.from_pretrained("THUDM/LongWriter-llama3.1-8b", trust_remote_code=True)
            self.client = AutoModelForCausalLM.from_pretrained("THUDM/LongWriter-llama3.1-8b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
            self.client = self.client.eval()

        elif model_type == 'Phi-3.5-MoE':
            model = AutoModelForCausalLM.from_pretrained( 
                "microsoft/Phi-3.5-MoE-instruct",  
                device_map="auto",  
                torch_dtype="auto",  
                trust_remote_code=True,  
            ) 
            
            tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-MoE-instruct")
            pipe = pipeline( 
                    "text-generation", 
                    model=model, 
                    tokenizer=tokenizer, 
                ) 
            self.client = pipe

            
        
        
        else:
            self.client = None
            raise NotImplementedError
        
        super().__init__(sample_length=sample_length,
                     model=self.model_name,
                     model_param=self.system_prompt,
                     config_sampling=config_sampling,
                     seed=seed,
                     max_retries=max_retries,
                     timeout=timeout)
    
        
        
        

    def sample(self, prompt, sample_length = 256, seed = None, num_samples = 1, temperature = 0.5):
        if self.model_type == 'gpt' or self.model_type == 'openai-o1' or self.model_type == 'openai-o1-mini' or self.model_type == 'openai-o3-mini':
            response = self.client.chat.completions.create(
            model = self.model_name,
            max_tokens=sample_length,
            temperature= temperature,
            top_p = self._config_sampling['prob'],
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
                ]   
            )
            # print("sample method was triggered.")
        
            response_text = ''
            if len(response.choices) > 0:
                response_text = response.choices[0].message.content
            print(response_text)

        if self.model_type == 'deep-seek-api':

            response = self.client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=sample_length,
            temperature = temperature,
            top_p = self._config_sampling['prob'],
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
                ]   
            )

            response_text = ''
            if len(response.choices) > 0:
                response_text = response.choices[0].message.content



        if self.model_type == 'deep-seek-api-reasoning':
            response = self.client.chat.completions.create(
                model  = "deepseek-reasoner",
                max_tokens=sample_length,
                temperature= temperature,
                top_p = self._config_sampling['prob'],
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
                ],

            )

            response_text = ''
            if len(response.choices) > 0:
                response_text = response.choices[0].message.content

        if self.model_type == 'deep-seek-local' or self.model_type == 'qwen':
            response: ChatResponse = chat(model=self.ollama_model_name,
                messages=[
                {"role": "system", "content": self.system_prompt},
                {'role': 'user', 'content': prompt}
                ],
                temperature = temperature,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                top_p = self._config_sampling['prob'],
                max_tokens = sample_length,
                
                )
            

            

        elif self.model_type == 'gemini':
            messages = [
                {"role" : "user", "parts" : prompt}
            ]
            
            response = self.client.generate_content(messages, generation_config=genai.types.GenerationConfig(
                    # Only one candidate for now.
                    candidate_count=1,
                    stop_sequences=["x"],
                    max_output_tokens=sample_length,
                    temperature = temperature#self._config_sampling['temp'],
                ),
            )
            response_text = ''
            if len(response.text) > 0:
                response_text = response.text
                print(response_text)


        elif self.model_type == 'sonnet':
            response = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=sample_length,
            system = self.system_prompt,
            temperature = temperature,#self._config_sampling['temp'],
            top_p = self._config_sampling['prob'],
            messages=[
                {"role": "user", "content": prompt}
                ]
            )
            response_text = ''
            response_text = response.content[0].text
            print(response_text)
                            
                

        elif self.model_type == 'llama-3.1-70b-quant':

            messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}]
            gen_input = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to('cuda')

            generated_ids = self.client.generate(gen_input, max_new_tokens=sample_length, temperature=temperature, repetition_penalty=1.1, do_sample=True, eos_token_id = self.tokenizer.eos_token_id)
            response_text = self.tokenizer.decode(generated_ids[0][gen_input.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_space=True)
            print(response_text)

        elif self.model_type == 'llama-3.1-70b':
            message = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]

            completion = self.client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=message,
            temperature=temperature,
            max_tokens=sample_length,
            top_p=self._config_sampling['prob'],
            stream=False,
            stop=None,
            )

            response_text = completion.choices[0].message.content

        
        elif self.model_type == 'mixtral-8x7b':
            message = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]

            completion = self.client.chat.completions.create(
                model = self.model_name,
                messages=message,
                temperature=temperature,
                max_tokens=sample_length,
                top_p=self._config_sampling['prob'],
                stream=False,
                stop=None,
            )

            response_text = completion.choices[0].message.content


        elif self.model_type == 'mistral_large':
            messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}]

            response_text = self.client.invoke(messages, temperature = temperature,#self._config_sampling['temp'],
                                              max_tokens = sample_length, top_p = self._config_sampling['prob'])

        elif self.model_type == 'LongWriter-glm4-9b':
            response, history = self.client.chat(self.tokenizer, prompt, history=[],
                                                 max_new_tokens=32768, temperature = temperature,#self._config_sampling['temp'],
                                                 top_p = self._config_sampling['prob'],
                                                 top_k = 50, repetition_penalty=1)
            print(f'Response text: {response}')
            response_text = response

        elif self.model_type == 'LongWriter-llama3.1-8b':
            device = "cuda" if torch.cuda.is_available() else "cpu"
            prompt = f"[INST]{prompt}[/INST]"
            input = self.tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input.input_ids.shape[-1]
            output = self.client.generate(
                **input,
                max_new_tokens=32768,
                num_beams=1,
                do_sample=True,
                temperature = temperature,#self._config_sampling['temp'],
            )[0]
            response_text = self.tokenizer.decode(output[context_length:], skip_special_tokens=True)
            print(response_text)
            print(f'Response text: {response_text}')

        elif self.model_type == 'Phi-3.5-MoE':
            messages = [
                {'role':'system', 'content':self.system_prompt}, # NEEDS TESTING IF SYSTEM PROMPT IS ACTUALLY NECESSARY
                {'role':'user', 'content':prompt}
                       ]
            generation_args = { 
            "max_new_tokens": 2048, 
            "return_full_text": False, 
            "temperature": temperature, 
            "do_sample": False, }
        
            output = self.client(messages, **generation_args)
            response_text =  output[0]['generated_text']
        
        
        results = [LanguageResponse(text=response_text,
                                    text_length=len(response_text),
                                    prompt=prompt,
                                    prompt_length=len(prompt))]
        

        
        return results
    

    def process_script(self, script:str, use_plain_text:bool):
        """
        Script for creating the summary of the script which will be used for recommendations. 
        script: path_to_doc or document itself.
        Output: 
            summarized script
        """
        if use_plain_text == False:
            if 'pdf' in script:
                loader = PyPDFLoader(script, )
                document = loader.load()
                
            
            elif 'txt' in script:
                loader = TextLoader(script)
                document = loader.load()
                

            elif 'doc' in script or 'docx' in script:
                loader = Docx2txtLoader(script)
                document = loader.load()

        else:
            document = Document(page_content=script)
            print(type(document))


        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
        chain = load_summarize_chain(llm, chain_type="stuff")
        if use_plain_text == False:
            result = chain.invoke(document)
        else:
            result = chain.invoke([document])
        return result
    

    def give_recommendations(self, summary, prompt, temperature = 0.5, model_name = 'gpt-4o-mini', max_tokens = 1024):
        rec_prompt = create_prompt_for_rec(prompt, summary)
        
        output = send_request(prompt = rec_prompt, temperature=temperature, model_name= model_name, max_tokens = max_tokens)

        return output



class StoryGeneratorCustom:
    def __init__(self, dramatron_handler, story_generator, logline, dialog_sequence = True, analyze_toxicity = False, config = None):
        self.dramatron_handler = dramatron_handler
        self.story_generator = story_generator
        self.logline = logline
        self.dialog_sequence = dialog_sequence
        self.analyze_toxicity = analyze_toxicity
        if self.analyze_toxicity == True:
            self.tox_client = DramatronHandler(
                        model_type='gpt',
                        seed=DEFAULT_SEED,
                        sample_length=config['sample_length'],
                        max_retries=config['max_retries'],
                        config_sampling=config['sampling'],
                        timeout=TIMEOUT, 
                        config = config,)

    def toxic_processing(self, generated_element, temperature = 0):
        prompt = toxicity_template.format(text = generated_element)
        toxic_estimation_rates_str = self.tox_client.sample(prompt, 15, 0)[0].text
        toxic_estimation_nums = parse_toxic_rating(toxic_estimation_rates_str)
        print(f'Estimation of the toxic rate: {toxic_estimation_nums}')
        if max(toxic_estimation_nums) > 0.55:
            regenerate_prompt = regenerate_template.format(text = generated_element)
            regenerated_element = self.tox_client.sample(regenerate_prompt, int(len(generated_element.split(' '))*1.5), temperature=temperature)
            return regenerated_element
        else:
            return generated_element
        

    def parse_storyboard_elements(self, storyboard, add_none = True):
        # Initialize empty lists for each element
        visuals = []
        text = []
        characters = []

        # Parse Visuals
        visual_pattern = re.compile(r'\*\*Visuals\*\*:(.*?)(?=\n-|\n\n|\Z)', re.DOTALL)
        visual_matches = visual_pattern.findall(storyboard)
        visuals = [v.strip() for v in visual_matches]

        # Parse Text
        text_pattern = re.compile(r'\*\*Text\*\*:(.*?)(?=\n-|\n\n|\Z)', re.DOTALL)
        text_matches = text_pattern.findall(storyboard)
        text = [t.strip().strip('"') for t in text_matches ] # if t.strip() != "None"

        # Parse Characters
        character_pattern = re.compile(r'\*\*Characters\*\*:(.*?)(?=\n\n|\Z)', re.DOTALL)
        character_matches = character_pattern.findall(storyboard)
        for match in character_matches:
            if add_none:
                chars = [c.strip() for c in match.split(',')]
            else:
                chars = [c.strip() for c in match.split(',') if c.strip() != "None" and c.strip() != "None visible"]
            characters.extend(chars)

        # Remove duplicates from characters list
        characters = list(dict.fromkeys(characters))

        return visuals, text, characters
        

    def get_storyboards(self):
        list_of_chunks = self.split_script_text_in_chunks()
        list_of_boards = self.get_list_of_story_boards(list_of_chunks)
        gen_vis_list = []
        gen_text_list = []
        gen_char_list = []
        gen_list_of_dicts  = []
        for item in list_of_boards:
            visuals, text, list_of_used_chracters = self.parse_storyboard_elements(item)
            gen_vis_list.append(visuals)
            gen_text_list.append(text)
            gen_char_list.append(list_of_used_chracters)

            dict_for_generation = {txt : visual for txt, visual in zip(text, visuals) }
            gen_list_of_dicts.append(dict_for_generation)

        return gen_list_of_dicts, gen_vis_list, gen_text_list, gen_char_list
        

    def get_descriptions(self, el_type = 'character'):
        dict_of_descriptions = {}
        if el_type == 'places':
            for item in self.places.values():
                dict_of_descriptions[item.name] = item.description
        
        elif el_type == 'characters':
            for char_name, char_desc in self.characters[0].items():
                dict_of_descriptions[char_name] = char_desc

                
        
        else:
            raise NotImplementedError(f'Element type is not supported. Provided element type: {el_type}. Supporterd elements types: [characters, places]')
        
        return dict_of_descriptions

    def create_char_str(self):
        list_of_names = list(self.characters[0].keys())
        list_of_descriptions = list(self.characters[0].values())

        character_str = ''
        for name, description in zip(list_of_names, list_of_descriptions):
            character_str += f'{name} - {description}\n'
        
        return character_str

    def get_story(self):
        self.story_generator._title = self.title
        self.story_generator._characters = self.characters
        self.story_generator._scenes = self.scenes
        self.story_generator._places = self.places
        self.story_generator._dialogs = self.list_of_dialogs
        story = self.story_generator.get_story()
        script_text = render_story(story)
        
        return script_text
    
    def split_script_text_in_chunks(self):
        list_of_chunks = []
        for idx_to_add in range(len(self.scenes[0])):
            chunk_to_add = """"""
            chunk_to_add += 'The script is based on the storyline: \n' + self.logline + '\n\n'
            
            character_desc = ''
            for item in self.characters:
                for key, value in item.items():
                    character_desc += f'{key}: {value}\n\n'
            
            element = self.scenes[0][idx_to_add]
            scene_description = f'Scene: {idx_to_add+1}\nPlace: {element.place}\nPlot element: {element.plot_element}\nBeat: {element.beat}'

            value = self.places[element.place]
            place_description =f'{value.name} - Scene {idx_to_add+1}\n\n{value.description}'    
            dialog_to_add = self.list_of_dialogs[idx_to_add]
            
        
            chunk_to_add += character_desc
            chunk_to_add += '\n' + scene_description + '\n'
            chunk_to_add += '\n' + place_description 
            chunk_to_add += '\n\n\n' + dialog_to_add
            list_of_chunks.append(chunk_to_add)

        return list_of_chunks
    
    def get_list_of_story_boards(self, list_of_chunks):
        list_of_storyboards = []
        for chunk in list_of_chunks:
            storyboards_prompt = storyboards_prompt_template.format(script_text = chunk)
            output = self.dramatron_handler.sample(storyboards_prompt,sample_length = 4096)
            output = output[0].text
            list_of_storyboards.append(output)

        return list_of_storyboards
    
    def parse_img2txt_prompts(self, list_of_storyboards):
        prompts_dictionary = dict()
        #TO DO: Create function that retrieves the visual and text parts
        pass

    def generate(self, content_type, sample_length = 2048, initial_character_data = None, initial_places_data = None, initial_scenes_data = None, temperature = 0.5):
        print('Custom Story Generator was called!')
        if content_type == 'title':
            prompt = title_format_template.format(logline = self.logline)
            gen_title = self.dramatron_handler.sample(prompt, sample_length, temperature)
            gen_title = gen_title[0].text
            if self.analyze_toxicity == True:
                gen_title = self.toxic_processing(gen_title)

            if '"' in gen_title:
                gen_title = gen_title.replace('"', '')
            self.title = Title.from_string(TITLE_ELEMENT + gen_title)
            
            
        if content_type == 'characters':
            prompt = characters_prompt_template.format(logline = self.logline, title = self.title)
            # print(f'Initial_character_data: {initial_character_data}')
            if initial_character_data is not None:
                prompt = characters_prompt_template_with_initial_data.format(logline = self.logline, title = self.title, char_init_data = initial_character_data)
                # print(f'Prompt with initial character information: {prompt}')
            generated_characters = self.dramatron_handler.sample(prompt, sample_length, temperature)
            generated_characters = generated_characters[0].text
            if self.analyze_toxicity == True:
                generated_characters = self.toxic_processing(generated_characters)

            self.characters = Characters.from_string(generated_characters)
            self.characters_str = self.create_char_str()
            
        if content_type == 'scenes':
            if initial_scenes_data is None:
                prompt = scenes_prompt_template.format(logline = self.logline, title = self.title,
                                                       character_str = self.characters_str, example = example_of_scenes)
            else:
                prompt = scenes_prompt_template_with_II.format(logline = self.logline, title = self.title,
                                                               character_str = self.characters_str, example = example_of_scenes, 
                                                               init_scenes_data = initial_scenes_data)
            generated_scenes = self.dramatron_handler.sample(prompt, sample_length, temperature)
            generated_scenes = generated_scenes[0].text
            if self.analyze_toxicity == True:
                generated_scenes = self.toxic_processing(generated_scenes)

            scenes = Scenes.from_string(generated_scenes)
            self.scenes = scenes
            self.list_of_place_name = []
            for item in scenes[0]:
                place_name = item.place
                self.list_of_place_name.append(place_name)
            
            
            
            
        if content_type == 'places':
            prompts_list = []
            print(f'INITIAL_PLACES_DATA: {initial_places_data}')
            if initial_places_data is None:
                for place_name in self.list_of_place_name:
                    prompt_to_add = place_prompt_template.format(logline=self.logline, place_name=place_name)
                    prompts_list.append(prompt_to_add)
            elif initial_places_data is not None:
                #check type
                
                if str(type(initial_places_data)) != "<class 'dict'>":
                    raise NotImplementedError(f"Initial data about places should be provided as a dictionary. Current format: {str(type(initial_places_data))}")
                else:
                    for place_name in self.list_of_place_name:
                        prompt_to_add = place_prompt_template_with_II.format(logline = self.logline, place_name = place_name, init_place_desc = initial_places_data[place_name])
                        print(f'PLACES PROMPT: {prompt_to_add}')
                        prompts_list.append(prompt_to_add)
                    

            places = {}
            for ind, prompt in enumerate(prompts_list):
                response_place = self.dramatron_handler.sample(prompt, sample_length, temperature)
                response_place = response_place[0].text
                response_place = response_place.replace('**END**', '')
                if self.analyze_toxicity == True:
                    response_place = self.toxic_processing(response_place)

                place = Place(self.list_of_place_name[ind], response_place)
                
                places[self.list_of_place_name[ind]] = place

            self.places = places

            

        if content_type == 'dialog':
            if self.dialog_sequence == False:
                list_of_dialogs = []

                list_of_dialogs_prompts = []
                for ind, item in enumerate(self.scenes[0]):
                    dialog_prompt = dilog_prompt_template.format(logline=self.logline, character_descriptions=self.characters_str,
                                                             scene = item.to_string(), place_description = self.places[item.place].description)
                    list_of_dialogs_prompts.append(dialog_prompt)

                for dialog_prompt in list_of_dialogs_prompts:
                    response_dialog = self.dramatron_handler.sample(dialog_prompt, sample_length, temperature)
                    response_dialog = response_dialog[0].text
                    if self.analyze_toxicity == True:
                        response_dialog = self.toxic_processing(response_dialog)
                    # print(f"RESPONSE OF THE MODEL: {response_dialog}")
                    list_of_dialogs.append(response_dialog)

                self.list_of_dialogs = list_of_dialogs

            elif self.dialog_sequence == True:
                list_of_dialogs = []
                item = self.scenes[0][0]
                first_prompt = dilog_prompt_template.format(logline = self.logline, character_descriptions=self.characters_str,
                                                             scene = item.to_string(), place_description = self.places[item.place].description)
                
                first_response_dialog = self.dramatron_handler.sample(first_prompt, sample_length, temperature)
                first_response_dialog = first_response_dialog[0].text
                list_of_dialogs.append(first_response_dialog)

                for ind in range(1, len(self.scenes[0])):
                    item = self.scenes[0][ind]
                    if ind == 1:
                        prompt = dialog_prompt_template_sequence.format(logline = self.logline, character_descriptions=self.characters_str,
                                                                 scene = item.to_string(), place_description = self.places[item.place].description, 
                                                                 previous_dialog = first_response_dialog)
                        
                        response_dialog = self.dramatron_handler.sample(prompt, sample_length, temperature)
                        response_dialog = response_dialog[0].text
                        if self.analyze_toxicity == True:
                            response_dialog = self.toxic_processing(response_dialog)
                        list_of_dialogs.append(response_dialog)
                    else:
                        prompt = dialog_prompt_template_sequence.format(logline = self.logline, character_descriptions=self.characters_str,
                                                                 scene = item.to_string(), place_description = self.places[item.place].description, 
                                                                 previous_dialog = response_dialog)
                        response_dialog = self.dramatron_handler.sample(prompt, sample_length, temperature)
                        if self.analyze_toxicity == True:
                            response_dialog = self.toxic_processing(response_dialog)
                        response_dialog = response_dialog[0].text
                        list_of_dialogs.append(response_dialog)

                self.list_of_dialogs = list_of_dialogs

        


        

    





