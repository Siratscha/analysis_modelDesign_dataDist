import torch

from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel,PNDMScheduler
from transformers import CLIPTokenizer
from transformers import CLIPImageProcessor


import os

model_path = "/work/srankl/thesis/development/modelDesign_bias_CXR/diffusers/roentGen_sd_lessnF"

unet = UNet2DConditionModel.from_pretrained(model_path + "/checkpoint-9500/unet")

vae = AutoencoderKL.from_pretrained(model_path + "/vae")
                   

tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")

feature_ex = CLIPImageProcessor.from_pretrained(model_path + "/feature_extractor")
noise_scheduler = PNDMScheduler.from_pretrained(model_path + "/scheduler")

pipe = StableDiffusionPipeline.from_pretrained(model_path, revision="fp16", safety_checker=None, torch_dtype=torch.float16, ).to("cuda:4")

def store_img(num_samples,gender_list, label_list):
    cwd = "/work/srankl/thesis/modelDesign_bias_CXR/images/SD_examples"
    for i in range(num_samples):
        for gender in gender_list:
            for label in label_list:
                combination = (gender,label)
                #gender_list.append(combination[0][0].upper())
                prompt = combination[0] +" - " + combination[1]
                image = pipe(prompt,num_inference_steps=75,guidance_scale=4.0,).images[0]
                image_name = prompt +str(i)

            
                
                path = os.path.join(cwd,image_name) + '.jpg'
                
                # Create the directories if they don't exist
                os.makedirs(os.path.dirname(path), exist_ok=True)
                image.save(path)
    
label_list = ['Edema','Cardiomegaly','Support Devices','Atelectasis','Pleural Effusion','Lung Opacity'] 
gender_list = ['female', 'male']

store_img(4,gender_list,label_list)