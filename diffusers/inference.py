import torch

from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel,PNDMScheduler
from transformers import AutoTokenizer, AutoModel, CLIPTokenizerFast, CLIPTokenizer
from transformers import CLIPImageProcessor

import random
from collections import defaultdict
import pandas as pd
import os

import string

model_path = "/work/srankl/thesis/development/modelDesign_bias_CXR/diffusers/roentGen_sd_lessnF"

unet = UNet2DConditionModel.from_pretrained(model_path + "/checkpoint-9500/unet")

vae = AutoencoderKL.from_pretrained(model_path + "/vae")
                   

tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")

feature_ex = CLIPImageProcessor.from_pretrained(model_path + "/feature_extractor")
noise_scheduler = PNDMScheduler.from_pretrained(model_path + "/scheduler")

pipe = StableDiffusionPipeline.from_pretrained(model_path, revision="fp16", safety_checker=None, torch_dtype=torch.float16, ).to("cuda:0")


def aggregate(df,labels, byGender):
    # Initialize an empty dictionary to store the aggregation functions
    aggregation_functions = {}

    # Iterate over the columns and add them to the aggregation functions dictionary
    for column in labels:
        aggregation_functions[column] = 'sum'
    if byGender:
        # Perform the dynamic aggregation
        result = df.groupby(['gender']).agg(aggregation_functions)
    else:
        result = pd.DataFrame(df.agg(aggregation_functions)).T

    return result

def draw_x_times(list, x):
    result_list = [random.choice(list) for _ in range(x)]
    return result_list


label_subset = ['Edema','Cardiomegaly','Support Devices','Atelectasis','Pleural Effusion','Lung Opacity'] 
gender_tokens = ['female', 'male']

def determine_num_prompts(gender_switch,label_switch, num_samples,label_subset,gender_tokens, data_subset):
    if not label_switch and not gender_switch:
        label_subset = ['Edema','Cardiomegaly','Lung Opacity','Atelectasis']

    if label_switch and not gender_switch:
        label_aggregation = aggregate(data_subset,label_subset,False)
        limit = num_samples
        num_samples = 0
        random_labels = [] 
        for label_index in label_subset:
            label_left = label_aggregation[label_index]
            num_samples_dis = int(limit - label_left )
            #num_samples_dis = 10
            if num_samples_dis > 0:
                random_labels.extend([label_index]  * num_samples_dis)
                num_samples += num_samples_dis


        random_genders = draw_x_times(gender_tokens, num_samples) 
        return num_samples, random_genders, random_labels
    if gender_switch and not label_switch:
        random_labels = draw_x_times(label_subset, num_samples)
        # switch for male or female
        random_genders = ['female'] * num_samples
        return num_samples, random_genders, random_labels
    #if label_switch and gender_switch:
    label_aggregation = aggregate(data_subset,label_subset,True)
    gender_labels = ['F', 'M']
    random_genders = []
    random_labels = []
    limit = num_samples
    num_samples = 0
    for i, gender in enumerate(gender_labels):
        for label_index in label_subset:
            gender_left = int(label_aggregation.loc[gender,label_index])
            num_samples_gend_dis = int(limit - gender_left )
            #num_samples_gend_dis = 10
            if num_samples_gend_dis > 0:
                random_genders.extend([gender_tokens[i]] * num_samples_gend_dis)
                random_labels.extend([label_index]  * num_samples_gend_dis)
                num_samples += num_samples_gend_dis
    print("Generating ", num_samples, " images!")
    return num_samples, random_genders, random_labels

def generate_random_string(num_samples):
    random_strings = []
    for i in range(num_samples):

        chars_per_group = 8
        num_groups = 5

        random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=chars_per_group*num_groups))

        # Insert dashes after every 8 characters (except the last group)
        random_string_with_dashes = '-'.join([random_string[i:i+chars_per_group] for i in range(0, chars_per_group*num_groups, chars_per_group)])
        random_strings.append(random_string_with_dashes)

    return random_strings

def store_img(folder,num_samples,gender_list, label_list, dicom_ids):
    #cwd = r"C:\Users\rankl\Documents\uni\Thesis\Development\modelDesign_bias_CXR\data\MIMICCXR"
    cwd = "/work/srankl/thesis/modelDesign_bias_CXR/data/MIMICCXR/physionet.org/files/mimic-cxr-jpg/2.0.0/files"
    for i in range(num_samples):
        combination = (gender_list[i], label_list[i])
        #gender_list.append(combination[0][0].upper())
        prompt = combination[0] +" - " + combination[1]
        image = pipe(prompt,num_inference_steps=75,guidance_scale=4.0,).images[0]
        patient = "p"+ str(folder[:2])
        patient_complete = "p"+ str(folder)
        subject = "s" + str(folder)
    
        
        path = os.path.join(cwd, patient, patient_complete, subject, dicom_ids[i]) + '.jpg'
        
        # Create the directories if they don't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path)

def store_sample(folder,num_samples, dicom_ids, gender_list, label_list, label_subset):
    
    folder_list = [folder] * num_samples
    df = pd.DataFrame({'subject_id': folder_list, 'study_id': folder_list})
    #df['subject_id'] = df['study_id'] = folder
    
    first_letters_list = [name[0].upper() for name in gender_list]
    df['gender'] = first_letters_list
    df['dicom_id'] = dicom_ids
    df['split'] = 'train'

    df_labels = pd.DataFrame({'Labels': label_list})
    one_hot_encoded = pd.get_dummies(df_labels['Labels'])
    columns_not_in_df  = set(label_subset) - set(one_hot_encoded.columns) 
    for column in columns_not_in_df:
        one_hot_encoded[column] = 0
    one_hot_encoded = one_hot_encoded[label_subset]

    combined_df = pd.concat([df, one_hot_encoded], axis=1)

    return combined_df

#samples_folder = ['20000010','20030000','20050000', '2060000', '20070000'] # , '20020000', '20030000'
#samples_folder = ['21005000', '21010000', '21020000']
#samples_folder = ['22023000', '22026000', '22030000', '22035000', '22040000', '22045000']

folder = '22030000' #samples_folder[1]
num_samples = int(folder[-5:])
data_subset = pd.read_csv(r"/work/srankl/thesis/development/modelDesign_bias_CXR/data/MIMICCXR/22030000.csv")#data_subset.csv
training_data = data_subset.loc[data_subset["split"] == "train"]
num_samples, random_genders, random_labels = determine_num_prompts(gender_switch = True,label_switch = True, num_samples = num_samples, label_subset=label_subset,gender_tokens=gender_tokens, data_subset=training_data)
x_ray_files = generate_random_string(num_samples)
output_df = store_sample(folder=folder,num_samples=num_samples,dicom_ids=x_ray_files, gender_list=random_genders,label_list=random_labels, label_subset=label_subset)

#pd.read_csv(r"C:\Users\rankl\Documents\uni\Thesis\Development\modelDesign_bias_CXR\data\MIMICCXR\data_subset.csv")

if not output_df.columns.equals(data_subset.columns):
    print("Warning: The column order does not match between output_df and data_subset.")

# Append rows of output_df to data_subset
combined_df = pd.concat([data_subset, output_df], ignore_index=True, verify_integrity=True)

# If you want to overwrite data_subset with the combined DataFrame:
# data_subset = combined_df

# If you want to save the combined DataFrame to a new CSV file:
cwd = r'/work/srankl/thesis/development/modelDesign_bias_CXR/data/MIMICCXR/'
path = os.path.join(cwd,folder)  + ".csv"
combined_df.to_csv(path, index=False)

store_img(folder=folder, num_samples=num_samples,gender_list=random_genders,label_list=random_labels,dicom_ids=x_ray_files)
