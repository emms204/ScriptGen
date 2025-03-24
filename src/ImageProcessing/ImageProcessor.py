import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from openai import OpenAI
from dotenv import load_dotenv
import os
import anthropic
import base64
from tqdm import tqdm

load_dotenv()




class FaceProcessor:
    def __init__(self, portrait_threshold = 0.18):
        base_options = python.BaseOptions(model_asset_path='detector.tflite')
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.detector = vision.FaceDetector.create_from_options(options)
        self.portrait_threshold = portrait_threshold


    def detect_faces(self, path_to_img):

        output_bboxes = []
        img = mp.Image.create_from_file(path_to_img)
        detection_result = self.detector.detect(img)
        
        for item in detection_result.detections:
            output_bboxes.append(item.bounding_box)
        
        if len(output_bboxes) == 0:
            error_message = "Faces cannot be detected!"
            print(error_message)
            return error_message, output_bboxes
        
        error_message = 'OK!'
        print(error_message)
        return error_message, output_bboxes

        


    def crop_faces(self, path_to_img, input_folder_path, path_to_output_img = None, to_save = True):
        path_to_img = os.path.join(input_folder_path, path_to_img)
        
        base_img = cv2.imread(path_to_img)
        base_height, base_width,  _ =  base_img.shape
        base_area = base_height * base_width
        message, coords_list = self.detect_faces(path_to_img)
        for i, bbox in enumerate(coords_list):
            x1 = bbox.origin_x
            y1 = bbox.origin_y
            x2 = x1 + bbox.width
            y2 = y1 + bbox.height
            cropped_img = base_img[y1:y2, x1:x2, :]
            # cv2.imshow('window', cropped_img)
            # cv2.waitKey(0)
            cropped_height, cropped_width, _ = cropped_img.shape
            cropped_area = cropped_height * cropped_width

            area_ratio = round(cropped_area/base_area, 3)
            # print(area_ratio)
            if area_ratio > self.portrait_threshold:
                message = 'Picture is okay!'
                if to_save  == True:
                    if path_to_output_img is not None:
                        cv2.imwrite(path_to_output_img, base_img)
                    else:
                        raise AssertionError('User decided to save the image but did not provide path to the output image.')
                return message, base_img
            
            else:
                x1 = int(bbox.origin_x - bbox.width)  
                x2 = int(bbox.origin_x + 1.75 * bbox.width)
                y1 = int(bbox.origin_y - 0.5* bbox.height)
                if y1 < 0:
                    y1 = 0
                y2 = int(bbox.origin_y + 1.75 * bbox.height)
                # print(x1, x2, y1, y2)
                cropped_img = base_img[y1:y2, x1:x2, :]
                # cv2.imshow('window_2', cropped_img)
                # cv2.waitKey(0)
                cropped_height, cropped_width, _ = cropped_img.shape
                cropped_area = cropped_height * cropped_width
                area_ratio = round(cropped_area/base_area, 3)
                # print(area_ratio)
                
                message = 'Picture was cropped!'
                print(message)
                if to_save == True:                
                    if path_to_output_img is not None:
                        print(f'Provided path: {path_to_output_img}')
                        cv2.imwrite(path_to_output_img, cropped_img)
                    else:
                        raise AssertionError('User decided to save the image but did not provide path to the output image.')
                return message, cropped_img
                



class Img2Text:
    def __init__(self, llm_name):
        self.llm_name = llm_name
        if self.llm_name == 'gpt':
            self.client = OpenAI()
        elif self.llm_name == 'sonnet':
            self.client = anthropic.Anthropic()
        if self.llm_name not in ['gpt', 'sonnet']:
            self.client = OpenAI()


    def encode_img(self, path_to_img):
        with open(path_to_img, 'rb') as image_file:
            image_data = image_file.read()
            bas64_encoded_data = base64_encoded_data = base64.b64encode(image_data)
            base64_encoded_string = base64_encoded_data.decode('utf-8')

        return base64_encoded_string

    def convert(self, img_path, temperature = 0, input_type = 'character'):

        encoded_img = self.encode_img(img_path)
        if input_type == 'character':
            prompt = 'Describe the character depicted on the image.'
        elif input_type == 'place':
            prompt = 'Describe the place depicted on the image.'
        
      
        if self.llm_name == 'gpt':
            messages = [
            {'role':'user', 'content': [
                {'type':'text', 'text':prompt},
                {'type':'image_url', 'image_url':{'url':f"data:image/jpeg;base64,{encoded_img}"}}
                ]
            }
            ]
            response = self.client.chat.completions.create(
                model='gpt-4o', 
                messages=messages,
                temperature=temperature, 
                max_tokens=1024,)
            response_text = response.choices[0].message.content

        if self.llm_name == 'sonnet':
            messages = [
                {
                "role": "user",
                "content": [
                    {"type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": encoded_img,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]}]

            model = 'claude-3-5-sonnet-20240620'
            response = self.client.messages.create(
                model = model,
                max_tokens=1024,
                temperature=temperature,
                messages = messages
            )
            response_text = response.choices[0].message.content

        return response_text


        
                
                
        