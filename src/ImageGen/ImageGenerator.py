
from src.TextGen.prompts import desc_to_prompt_template
import torch
from diffusers import FluxPipeline
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers import StableDiffusion3Pipeline
from diffusers import StableDiffusionLatentUpscalePipeline
import transformers
from huggingface_hub import login
import os
import anthropic
from openai import OpenAI
from diffusers import StableDiffusionPipeline
from FaceDetailer import FaceDetailer




class ImageGenerator:
    def __init__(self, model_type, model_name, llm_name, temperature, allow_cpu_offload = True, token = None):

        self.frequency_penalty = 0.2
        self.presence_penalty = 0.2
        self.temperature = temperature
        self.model_type = model_type
        self.model_name = model_name
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        login(token = token)
        self.llm_name = llm_name

        if llm_name == 'gpt-4o' or llm_name == 'gpt-4o-mini':
            self.LLM = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
        elif llm_name == 'sonnet':
            self.LLM = anthropic.Anthropic(api_key=os.getenv("SONNET_API_KEY"))

        if self.model_type == 'Flux':
            if model_name == 'dev':
                model_id ="black-forest-labs/FLUX.1-dev"
                self.model = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
                if allow_cpu_offload:
                    self.model.enable_model_cpu_offload()

                
            elif model_name == 'schnell':
                model_id = "black-forest-labs/FLUX.1-schnell"
                self.model = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
                if allow_cpu_offload:
                    self.model.enable_model_cpu_offload()


        elif self.model_type == 'StableDiffusion':
            if self.model_name == 'stable-diffusion-xl':
                model_id = "stabilityai/stable-diffusion-xl-base-1.0"
                self.model = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
                if allow_cpu_offload:
                    self.model.enable_model_cpu_offload()
                self.model.to(self.device)
            else:
                try:
                    print('Loading the hugging face checkpoint.')
                    self.model = StableDiffusionXLImg2ImgPipeline.from_pretrained(self.model_name, torch_dtype = torch.float16)
                    if allow_cpu_offload:
                        self.model.enable_model_cpu_offload()
                    self.model.to(self.device)

                except Exception as e:
                    print(f'Loading failed due to following reason:\n{e} ')


        else:
            raise NotImplementedError(f'Not supported model type: {self.model_type}')
        

    def desc2prompt(self, description, temperature):
        dess_conv_prompt = desc_to_prompt_template.format(text = description)
        if 'gpt' in self.llm_name:
            response = self.LLM.chat.completions.create(
            model = self.llm_name,
            max_tokens = 700,
            temperature= temperature,#self._config_sampling['temp'],
            top_p = 0.8,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            messages=[
                {"role": "user", "content": dess_conv_prompt}
                ]   
            )
    
            response_text = ''
            if len(response.choices) > 0:
                response_text = response.choices[0].message.content
            return response_text

        elif 'sonnet' in self.llm_name :
            response = self.LLM.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=700,
            temperature = temperature,#self._config_sampling['temp'],
            top_p = 0.8,
            messages=[
                {"role": "user", "content": dess_conv_prompt}
                ]
            )

            response_text = ''
            response_text = response.content[0].text
            print(response_text)       

        return response_text
    

    def desc2img(self, description, temperature, num_inference_steps, guidance_scale, style_of_generation):
        prompt_for_img = self.desc2prompt(description, temperature)
        prompt_for_img = f"{style_of_generation}.  {prompt_for_img}"
        seed = 42
        print(self.model_name)
        if self.model_type == 'Flux':
            if self.model_name == 'schnell':
                image = self.model(
                        prompt_for_img,
                        output_type="pil",
                        num_inference_steps=num_inference_steps, #use a larger number if you are using [dev]
                        generator=torch.Generator("cpu").manual_seed(seed)
                    ).images[0]

            elif self.model_name == 'dev':
                image = self.model(
                        prompt_for_img,
                        height=1024,
                        width=1024,
                        guidance_scale=guidance_scale,
                        output_type="pil",
                        num_inference_steps=num_inference_steps,
                        max_sequence_length=512,
                        generator=torch.Generator("cpu").manual_seed(seed)
                    ).images[0]



        elif self.model_type == 'StableDiffusion':
            image = self.model(
                prompt=prompt_for_img,
                height=1024,
                width=1024,
                guidance_scale=guidance_scale,
                output_type="pil",
                num_inference_steps=num_inference_steps,
                generator=torch.Generator("cpu").manual_seed(seed)

            )

        else:
            raise NotImplementedError(f'Provided model name and/or is not supported. Model-type : {self.model_type}, Model-name : {self.model_name}')

        return image
    
    def generate_images(self, dict_of_descriptions, temperature = 0.5, num_inference_steps  = 30, guidance_scale=3.5, style_of_generation = ''):
        dict_of_images = {}
        for key, value in dict_of_descriptions.items():
            image = self.desc2img(value, temperature, num_inference_steps, guidance_scale, style_of_generation)
            dict_of_images[key] = [image, value]

        return dict_of_images
    




class ImageGeneratorAdvanced:
    def __init__(self) -> None:
        pass



class ImageUpscaleAdvanced:
    def __init__(self, base_model_type = 'SD', seed = 33) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if base_model_type == 'SD':
            #model_id = "stabilityai/stable-diffusion-xl-base-1.0" 
            model_id = "stabilityai/stable-diffusion-3.5-large"
            self.init_pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16
            )
            

        elif base_model_type == 'FLUX':
            flux_model_id = "black-forest-labs/FLUX.1-dev"

            self.init_pipe = FluxPipeline.from_pretrained(
                flux_model_id, 
                torch_dtype = torch.bfloat16
            )
        
        self.init_pipe.enable_model_cpu_offload()

        upscaler_model_id = "stabilityai/sd-x2-latent-upscaler"
        self.upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
            upscaler_model_id,
            torch_dtype=torch.float16
        )
        self.generator = torch.manual_seed(seed)
        sd_15_id = "sd-legacy/stable-diffusion-v1-5"

        self.sd_15 = StableDiffusionPipeline(
            sd_15_id,
            torch_dtype = torch.float16
        )

        self.face_detailer = Face
        



    def initial_generation(self, prompt, return_low_res = False, guidance_scale = 3.5, height = 768, width = 1360, num_inference_steps = 50):
        self.init_pipe.to(self.device)

        if not return_low_res:
            gen_img = self.init_pipe(
                prompt = prompt, 
                guidance_scale = guidance_scale,
                height = height, 
                width = width,
                num_inference_steps = num_inference_steps
            ).images[0]
            self.init_pipe.to('cpu')
            return gen_img
        else:
            gen_low_res = self.init_pipe(
                prompt = prompt, 
                generator = self.generator, 
                output_type = 'latent',
            ).images

            self.init_pipe.to('cpu')
            return gen_low_res 

    def latent_upscale(self, gen_low_res, initial_prompt, num_inference_steps = 20, guidance_scale = 0, ):
        upscaled_img = self.latent_upscale(
            prompt = initial_prompt, 
            image = gen_low_res, 
            guidance_scale = guidance_scale, 
            generator = self.generator, 
            num_inference_steps = num_inference_steps,
        ).images[0]

        return upscaled_img
    

    # def final_run(self, prompt, num_inference_steps = 20, guidance_scale = 0):

    






