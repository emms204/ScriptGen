import os
from PIL import Image, ImageOps
from faster_whisper import WhisperModel
from src.ImageProcessing.ImageProcessor import FaceProcessor
import shutil
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import contextlib
import wave

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Load your images
#Replace it with user input
folder_path = 'test_input_folder_for_face_grid'
output_path = 'pictures_for_face_grid'
image_paths = ["ref_img_2.jpg", "ref_img_2_copy.jpg", "ref_img_2_copy_2.jpg", "ref_img_2_copy_3.jpg", 'ref_img_2_copy_4.jpg']
list_of_names = ['Name_1', 'Name_2','Name_3', 'Name_4', 'Name_5']
# image_paths = [os.path.join(folder_path, image_path) for image_path in image_paths]



class VideoProcessor:
    def __init__(self):

        self.face_proc = FaceProcessor()
        self.placeholder_path = 'placeholder.png'

    def initial_preprocessing(self, list_of_initial_images_pathes, output_folder_name, input_folder_path):
        list_of_processed_images = []
        for i, file_path in enumerate(list_of_initial_images_pathes):   
            self.face_proc.crop_faces(path_to_img=file_path, input_folder_path=input_folder_path, path_to_output_img=os.path.join(output_folder_name, file_path), to_save=True)
            list_of_processed_images.append(os.path.join(output_folder_name, file_path))
        
        print(list_of_processed_images)
        return list_of_processed_images
    
    def merge_audios(self, dict_with_audios):
        pass
        

    def create_grid(self, input_folder_path, list_of_initial_image_pathes,
                    list_of_characters_names, processed_folder_path, output_path = 'photo_grid_with_borders.jpg',
                    border_size = 5, border_color = 'black', size_of_the_cell = (200,200), font_size = 30):
        
        # define grid shape
        list_of_image_pathes = self.initial_preprocessing(list_of_initial_image_pathes, processed_folder_path, input_folder_path=input_folder_path)
        template_num = 0
        

        if len(list_of_image_pathes) %3 ==0:
            #no place holder
            second_num = int(len(list_of_image_pathes)/3)
            grid_size = (3, second_num)

        elif len(list_of_image_pathes) %3 != 0:
            second_num = len(list_of_image_pathes)/3
            first_part = str(second_num)[0]
            second_part = str(second_num)[2:4]
            second_num = int(first_part) + 1
            if second_part == '33':
                template_num = 2
            elif second_part == '66':
                template_num = 1
            
            for i in range(template_num):
                list_of_image_pathes.append(self.placeholder_path)
            
            grid_size = (3, second_num)

        images = [ImageOps.expand(Image.open(img).resize(size_of_the_cell), 
                          border=border_size, fill=border_color) for img in list_of_image_pathes]
        for i, img in enumerate(images):
            draw = ImageDraw.Draw(img)
            if i >= len(images) - template_num:
                print(i)
                continue
    
            name_to_write = list_of_characters_names[i] 
            
            position = (50, 170)
            font = ImageFont.load_default(size = font_size)
            color = (255, 255, 255)
            draw.text(position, name_to_write, fill=color, font=font)


        grid_img = Image.new('RGB', (
            grid_size[0] * (size_of_the_cell[0] + 2 * border_size), 
            grid_size[1] * (size_of_the_cell[1] + 2 * border_size)),
            color=border_color  # Background color for the grid (optional)
        )

        for index, img in enumerate(images):
            row = index // grid_size[0]
            col = index % grid_size[0]
            x_offset = col * (size_of_the_cell[0] + 2 * border_size)
            y_offset = row * (size_of_the_cell[1] + 2 * border_size)
            grid_img.paste(img, (x_offset, y_offset))
            
            
        grid_img.save(output_path)

        return grid_img, output_path
    
    def postprocess_wordtimings(self, old_word_timing):
        new_word_timings = []
        for i, item in enumerate(old_word_timing):
            item['word'] = item['word'][1:]
            item['index'] = i
            new_word_timings.append(item)
        
        return new_word_timings
    
    def generate_word_timings(self, audio_file):
        # Initialize Whisper model
        model_size = "base"
        model = WhisperModel(model_size, device="cpu")
        
        # Transcribe audio and extract word timings
        segments, _ = model.transcribe(audio_file, word_timestamps=True)
        
        # Prepare word timing information for each segment
        word_timings = []
        for segment in segments:
            for word in segment.words:
                word_timings.append({
                    'word': word.word,
                    'start': word.start,
                    'end': word.end
                })

        word_timings = self.postprocess_wordtimings(word_timings)
        return word_timings
    
    
    def wrap_text_with_indices(self, text, width):
        words = text.split()
        lines = []
        current_line = []
        word_idx = 0  
        for word in words:
            current_line.append((word, word_idx))
            word_idx += 1  
            
            if len(" ".join([w for w, idx in current_line])) > width:
                lines.append(current_line) 
                current_line = []  
        if current_line:
            lines.append(current_line)  
        return lines  
    
    def calculate_text_height(self, num_lines, line_height):
        return num_lines * line_height  # Total height of the text block

    def create_highlighted_text_image_with_scrolling(self, full_text, word_timings, current_time, video_width, video_height, font_size, line_wrap_width, video_duration, scroll_speed = 500):
        wrapped_text_lines = self.wrap_text_with_indices(full_text, line_wrap_width)
        

        img = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        
        # Set up font parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (255, 255, 255)
        highlight_color = (255, 69, 103)  
        font_scale = font_size / 40
        line_height = int(font_size * 1.5)
        

        total_text_height = len(wrapped_text_lines) * line_height
        
        max_scroll = max(0, total_text_height - video_height)
        max_scroll = - scroll_speed
        
        scroll_offset = int(max_scroll * (current_time / video_duration))  
        

        y0 = video_height - total_text_height + scroll_offset  
        
        # Iterate through each line and render text
        for line_idx, line in enumerate(wrapped_text_lines):
            y = y0 + line_idx * line_height  
            

            if y < -line_height:
                continue
            if y > video_height + line_height:
                continue

            x = video_width // 8  
            
            for word, word_idx in line:
                for word_info in word_timings:
                    word_text = word_info['word']
                    start_time = word_info['start']
                    end_time = word_info['end']
                    word_index = word_info['index']
                    
                    if word == word_text and word_idx == word_index and start_time <= current_time <= end_time + 0.05:
                        cv2.putText(img, word, (x, y), font, font_scale, highlight_color, 2, cv2.LINE_AA)
                        break
                else:
                    cv2.putText(img, word, (x, y), font, font_scale, font_color, 2, cv2.LINE_AA)
                
                x += int(font_scale * 20 * len(word)) + 10

        return img
    
    def merge_image_and_scrolling_text(self, static_image_path, full_text, word_timings, video_width, video_height, font_size, line_wrap_width, audio_file):
        static_img = cv2.imread(static_image_path)
        
        img_height, img_width, _ = static_img.shape
        
        
        if img_height > video_height or img_width > video_width:
        
            scale_factor = min(video_width / img_width, video_height / img_height)
            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)
            static_img = cv2.resize(static_img, (new_width, new_height))
        else:
            new_width, new_height = img_width, img_height

        text_area_x = new_width  
        text_area_y = new_height 

        with contextlib.closing(wave.open(audio_file, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate) 
        
       
        out = cv2.VideoWriter('highlighted_text_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 24, (video_width, video_height))  # Set FPS to 24
        fps = 24
        num_frames = int(duration * fps)
        
        # Generate frames for the duration of the video
        for frame_num in range(num_frames):  
            current_time = frame_num / 24.0  
            
            
            scrolling_text_img = self.create_highlighted_text_image_with_scrolling(
                full_text, word_timings, current_time, video_width - text_area_x, video_height, font_size, line_wrap_width, duration
            )
            
            combined_frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
            
            combined_frame[:new_height, :new_width] = static_img
            
            combined_frame[0:, text_area_x:] = scrolling_text_img
            
            out.write(combined_frame)
        
        out.release()

        import os
        os.system(f"ffmpeg -i highlighted_text_video.avi -i {audio_file} -c:v copy -c:a aac -strict experimental highlighted_text_video_with_audio.mp4")
        



if __name__ == '__main__':
    grd_proc = VideoProcessor()

    grd_proc.create_grid(list_of_initial_image_pathes=image_paths, input_folder_path=folder_path, list_of_characters_names=list_of_names,
                     processed_folder_path=output_path, )

    audio_file = 'download.wav'
    full_text = "Instead of the track, you notice five workers working on the track. You try to stop, but you can't. Your brakes don't work. You feel desperate because you know that if you crash into these five workers, they will all die. Let's assume you know that for sure. And so you feel helpless until you notice that there is, off to the right, a side track. And at the end of that track, there is one worker working on the track. Your steering wheel works. So you can turn the trolley car if you want to, onto the side track, killing the one, but sparing the five. Here's our first question."

    word_timings = grd_proc.generate_word_timings(audio_file)

    grd_proc.merge_image_and_scrolling_text('photo_grid_with_borders.jpg',full_text, word_timings, 1440, 720, 25, 50, audio_file)



       
