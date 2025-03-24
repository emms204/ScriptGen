import librosa
import torch
import torchaudio
import numpy as np
import random
from argparse import Namespace
import subprocess
import os
from TTS.api import TTS
from parler_tts import ParlerTTSForConditionalGeneration
from kokoro import KPipeline
import whisper

from transformers import AutoTokenizer
import soundfile as sf


from huggingface_hub import hf_hub_download



class Mars5TTS:
    def __init__(self):
        self.mars5, self.config_class = torch.hub.load('Camb-ai/mars5-tts', 'mars5_english', trust_repo=True)



    def generate(self, path_to_ref_audio, source_transcript, output_transcript, deep_clone = True):
        wav, sr = librosa.load(path_to_ref_audio, sr = self.mars5.sr, mono = True)
        wav = torch.from_numpy(wav)

        cfg = self.config_class(deep_clone = deep_clone, rep_penalty_window=100,
                      top_k=100, temperature=0.7, freq_penalty=3)
        
        ar_codes, output_audio = self.mars5.tts(output_transcript, wav, source_transcript, cfg = cfg)

        return output_audio.numpy()
    
    def save_gen_audios(self, dict_with_audio, output_folder):
        output_dict_with_pathes = {}
        for key, value in dict_with_audio.items():
            for ind, audio_arr in value.items():
                output_file_name = f'{key}_{ind}.wav'
                file_path = os.path.join(output_folder, output_file_name)


                sf.write(file_path, audio_arr, self.mars5.sr)
                output_dict_with_pathes[key][ind] = os.path.join(output_folder, output_file_name)
            
        
            

    def inference(self, dict_of_replics, ref_dict, output_folder):
        dict_with_audio = {}
        for key, value in dict_of_replics.items():
            #get the ref source audio path by the name
            ref_audio = ref_dict[key][0]
            ref_transcript = ref_dict[key][1]
            dict_with_audio[key] = {}
            for ind, line in value.items():
                dict_with_audio[key][ind] = self.generate(path_to_ref_audio = ref_audio, 
                                                          source_transcript = ref_transcript,
                                                          output_transcript = line)

        self.save_gen_audios(dict_with_audio, output_folder)
        return dict_with_audio        
            
                
        
class FishTTS:
    def __init__(self, base_checkpoint_path = "checkpoints/fish-speech-1.5",
                 path_to_audio_generator = "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
                  whisper_model_size = 'turbo') -> None:
        """Model for effective voice cloning."""
        
        self.base_checkpoin_path = base_checkpoint_path
        self.path_to_audio_generator = path_to_audio_generator
        
        self.whisper_model = whisper.load_model(whisper_model_size)


    def generate_audio(self, input_text, audio_for_cloning:str, output_path:str = 'generated_audio.wav'):
        text_from_audio = self.whisper_model.transcribe(audio_for_cloning)['text']
        #Encode input audio
        result = subprocess.run(f"fish_speech/models/vqgan/inference.py  -i {audio_for_cloning}  --checkpoint-path {self.path_to_audio_generator}",
                                 shell=True, text=True, capture_output=True)
        print(f'OUTPUT OF THE INPUT ENCODING PART: {result}')

        #Generate semantic tokens from text
        result = subprocess.run(
            f"fish_speech/models/text2semantic/inference.py \
            --text {input_text} \
            --prompt-text {text_from_audio} \
            --prompt-tokens fake.npy \
            --checkpoint-path {self.base_checkpoin_path} \
            --num-samples 2\
            --compile",
            shell = True,
            text = True, 
            capture_output=True
        )

        print(f'OUTPUT OF THE SEMANTIC TOKENS GENERATION : {result}')

        #Generate Audio:
        result = subprocess.run(
            f"fish_speech/models/vqgan/inference.py  -i codes_0.npy  --checkpoint-path {self.path_to_audio_generator}, --output-path {output_path}",
            shell=True, text=True, capture_output=True
        )
        
        print(f'Output of the audio generation block: {result}')

        
        data, sample_rate = sf.read(output_path)

        return data

        

        



class KokoroTTS:
    def __init__(self, target_language : str = 'American English', ) -> None:
        """
        Init method for the KokoroTTS model. 
        args:
            target_language : str
            Language of the output speech. Possible options = ['American English', 'British English',]
        """
        if target_language == 'American English':
            lang_code = 'a'
        elif target_language == 'British English':
            lang_code = 'b'
        else:
            raise ValueError('Unsupported language. List of supported languages: ["American English", "British English",]')

        self.pipeline = KPipeline(lang_code = lang_code)

    
    def generate_audio(self, input_text, target_voice, speed = 1, split_pattern = r'\n+'):
        """
        target_voices: 
            American English : 
                male voices: [am_adam - F+, am_echo - D, am_eric - D, am_fenrir - C+, am_liam - D, am_michael - C+, am_onyx  - D, am_puck - C+, am_santa - D-,]
                female voices: [af_heart - A, af_alloy - C, af_aoede C+, af_bella - A-, af_jessica - D, af_kore - C+, af_nicole - B-, af_nova - C, af_river - D, af_sarah - C+, af_sky - C-]

            British English : 
                male voices: [bm_daniel - D, bm_fable - C, bm_george - C, bm_lewis - D+]
                female voices: [bf_alice - D, bf_emma - B-, bf_isabella - C, bf_lily - D]

        """
        #TBD add small utterances filtering, to combine small ones and split very long ones.
        
        graphemes = list()
        phonemes = list()
        audios = list()

        generator = self.pipeline(
            input_text, target_voice,
            speed = speed, split_pattern = split_pattern
        )

        for i, (gs, ps, audio) in enumerate(generator):
            graphemes.append(gs)
            phonemes.append(ps)
            audios.append(audio)

        return audios, graphemes, phonemes


class CoquiTTS:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

    def inference(self, dict_of_replics, output_folder, dict_with_mapped_chars):
        """
        dict_with_mapped_chars: structure that is responsible for providing the reference_audios
        """
        output_dict_with_pathes = {}
        for key, value in dict_of_replics.items():
            for ind, line in value.items():
                key_to_write = key.replace(' ', '_')
                key_to_write = key_to_write.replace('__', '')

                self.model.tts_to_file(text = line,
                    speaker_wav = dict_with_mapped_chars[key],
                    language = 'en',
                    file_path = os.path.join(output_folder, f'{key_to_write}_{ind}.wav')
                )
                # output_dict_with_pathes[ind] = [line, os.path.join(output_folder, f'{key_to_write}_{ind}.wav')]
                output_dict_with_pathes[key][ind] = os.path.join(output_folder, f'{key_to_write}_{ind}.wav')

        return output_dict_with_pathes
    


class ParlerTTS:
    def __init__(self) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-large-v1").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-large-v1")

    def single_inference(self, prompt, description, file_path):
        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(self.device)
        prompt_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        generation = self.model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(file_path, audio_arr, self.model.config.sampling_rate)
        return audio_arr

    def general_inference(self, dict_of_replics, dict_with_mapped_chars, output_folder):
        """
        dict_with_mapped_chars = structure that is responsible for providing the reference_descriptions which will used for voice generation
        """
        dict_with_audio = {}
        dict_with_audio_pathes = {}

        for key, value in dict_of_replics.items():
            
            for ind, line in value.items():
                print(line)
                key_to_write = key.replace(" ", "_")
                key_to_write = key_to_write.replace('__', '')
                prompt = line
                description = dict_with_mapped_chars[key]
                output_file_name = os.path.join(output_folder, f'{key_to_write}_{ind}.wav')
                output = self.single_inference(prompt, description, output_file_name)
                dict_with_audio[key][ind] = output
                dict_with_audio_pathes[key][ind] = output_file_name

        return dict_with_audio_pathes


