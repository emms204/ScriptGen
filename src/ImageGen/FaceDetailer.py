import os
import torch
from PIL import Image, ImageFilter
import numpy as np
import cv2
from mediapipe import solutions
from ultralytics import YOLO
from folder_paths import base_path

import nodes

face_model_path = os.path.join(base_path, "models/dz_facedetailer/yolo/face_yolov8n.pt")
MASK_CONTROL = ["dilate", "erode", "disabled"]
MASK_TYPE = ["face", "box"]


def paste_numpy_images(target_image, source_image, x_min, x_max, y_min, y_max):
    target_image[y_min:y_max, x_min:x_max, :] = source_image
    return target_image

def image2nparray(image, BGR):
    narray = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return narray if BGR else narray[:, :, ::-1]

def set_mask(samples, mask):
    s = samples.copy()
    s["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
    return s



class FaceDetailer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {
                    "model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.SAMPLERS, ),
                    "scheduler": (comfy.samplers.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "vae": ("VAE",),
                    "mask_blur": ("INT", {"default": 0, "min": 0, "max": 100}),
                    "mask_type": (MASK_TYPE, ),
                    "mask_control": (MASK_CONTROL, ),
                    "dilate_mask_value": ("INT", {"default": 3, "min": 0, "max": 100}),
                    "erode_mask_value": ("INT", {"default": 3, "min": 0, "max": 100}),
                }
                }

    RETURN_TYPES = ("LATENT", "MASK",)
    FUNCTION = "detailer"
    CATEGORY = "face_detailer"

    def detailer(self, model, positive, negative, latent_image, seed = 11233431212, steps = 20, cfg = 7.0, sampler_name = 'ddim', scheduler = 'karras', denoise = 0.5, vae = None, mask_blur = 0, mask_type = 'mesh', mask_control = 'disabled', dilate_mask_value = 3, erode_mask_value = 3):
        # Decode latent for face detection.
        #tensor_img = vae.decode(latent_image["samples"])

        tensor_img = latent_image
        batch_size = tensor_img.shape[0]

        mask = Detection().detect_faces(tensor_img, batch_size, mask_type, mask_control, mask_blur, dilate_mask_value, erode_mask_value)
        true_latent_img = "" #vae.encode(latent_img) 
        latent_mask = set_mask(true_latent_img, mask)

        latent = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_mask, denoise=denoise)
        #vae decode
        #return decoded img
        return (latent[0], latent[0]["noise_mask"],)

class Detection:
    def __init__(self):
        pass

    def detect_faces(self, tensor_img, batch_size, mask_type, mask_control, mask_blur, mask_dilate, mask_erode):
        mask_imgs = []
        for i in range(batch_size):
            img = image2nparray(tensor_img[i], False)
            if mask_type == "box":
                final_mask = facebox_mask(img)
            else:
                final_mask = facemesh_mask(img)

            final_mask = self.mask_control(final_mask, mask_control, mask_blur, mask_dilate, mask_erode)
            final_mask = np.array(Image.fromarray(final_mask).getchannel('A')).astype(np.float32) / 255.0
            final_mask = torch.from_numpy(final_mask)
            mask_imgs.append(final_mask)
        final_mask = torch.stack(mask_imgs)
        return final_mask

    def mask_control(self, numpy_img, mask_control, mask_blur, mask_dilate, mask_erode):
        numpy_image = numpy_img.copy()
        if mask_control == "dilate":
            if mask_dilate > 0:
                numpy_image = self.dilate_mask(numpy_image, mask_dilate)
        elif mask_control == "erode":
            if mask_erode > 0:
                numpy_image = self.erode_mask(numpy_image, mask_erode)
        if mask_blur > 0:
            final_mask_image = Image.fromarray(numpy_image)
            blurred_mask_image = final_mask_image.filter(ImageFilter.GaussianBlur(radius=mask_blur))
            numpy_image = np.array(blurred_mask_image)
        return numpy_image

    def erode_mask(self, mask, dilate):
        kernel = np.ones((int(dilate), int(dilate)), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        return dilated_mask

    def dilate_mask(self, mask, erode):
        kernel = np.ones((int(erode), int(erode)), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        return eroded_mask

def facebox_mask(image):
    mask = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    face_model = YOLO(face_model_path)
    face_bbox = face_model(image)
    boxes = face_bbox[0].boxes
    for box in boxes.xyxy:
        x_min, y_min, x_max, y_max = box.tolist()
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        max_size = max(width, height)
        new_width = max_size
        new_height = max_size
        new_x_min = int(center_x - new_width / 2)
        new_y_min = int(center_y - new_height / 2)
        new_x_max = int(center_x + new_width / 2)
        new_y_max = int(center_y + new_height / 2)
        cv2.rectangle(mask, (new_x_min, new_y_min), (new_x_max, new_y_max), (0, 0, 0, 255), -1)
    return mask

def facemesh_mask(image):
    faces_mask = []
    mask = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    face_model = YOLO(face_model_path)
    face_bbox = face_model(image)
    boxes = face_bbox[0].boxes
    for box in boxes.xyxy:
        x_min, y_min, x_max, y_max = box.tolist()
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        max_size = max(width, height)
        new_width = max_size
        new_height = max_size
        new_x_min = int(center_x - new_width / 2)
        new_y_min = int(center_y - new_height / 2)
        new_x_max = int(center_x + new_width / 2)
        new_y_max = int(center_y + new_height / 2)
        face = image[new_y_min:new_y_max, new_x_min:new_x_max, :]
        mp_face_mesh = solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
        results = face_mesh.process(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                points = []
                for landmark in face_landmarks.landmark:
                    cx, cy = int(landmark.x * face.shape[1]), int(landmark.y * face.shape[0])
                    points.append([cx, cy])
                face_mask = np.zeros((face.shape[0], face.shape[1], 4), dtype=np.uint8)
                convex_hull = cv2.convexHull(np.array(points))
                cv2.fillConvexPoly(face_mask, convex_hull, (0, 0, 0, 255))
                faces_mask.append([face_mask, [new_x_min, new_x_max, new_y_min, new_y_max]])
    for face_mask in faces_mask:
        paste_numpy_images(mask, face_mask[0], face_mask[1][0], face_mask[1][1], face_mask[1][2], face_mask[1][3])
    return mask





