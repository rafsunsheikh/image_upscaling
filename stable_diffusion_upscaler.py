import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch
import cv2

# Load the model and Scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype = torch.float16)
# pipeline = pipeline.to("cuda")

def get_low_res_img(url, shape):
    response = requests.get(url)
    low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
    low_res_img = low_res_img.resize(shape)
    return low_res_img

url = "https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d?q=80&w=1000&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8aHVtYW58ZW58MHx8MHx8fDA%3D"
shape = (320, 400)
low_res_img = get_low_res_img(url, shape)
# low_res_img

prompt = "an aesthetic butterfly"
upscaled_image = pipeline(prompt = prompt, image = low_res_img).images[0]
# Save the image
cv2.imwrite("Upscaled_butterfly_stable_diffusion.png", upscaled_image)