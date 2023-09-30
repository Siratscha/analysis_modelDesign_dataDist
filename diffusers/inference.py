# This script is used during inference after fine tuning the text to image stable diffusion pipeline.
# It automatically detects how many samples are needed per class and group to reach a threshold (num_samples)
# For this it uses a csv of a dataset and aggregates number of labels per class and group and substracts that 
# frequency from the threshold to get the number of samples to be added

# users need to call **determine_num_prompts()** with gender_switch (bool), label_switch(bool) if they want to use specific label subsets, or
# if they want to generate images for one group (default = female) 
# gender_switch = True,label_switch = False uses default settings

import torch

# clone diffusers with git clone https://github.com/huggingface/diffusers.git
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel,PNDMScheduler
from transformers import AutoTokenizer, AutoModel, CLIPTokenizerFast, CLIPTokenizer
from transformers import CLIPImageProcessor

import random
from collections import defaultdict
import pandas as pd
import os

import string

model_path = <path to trained stable diffusion model>

unet = UNet2DConditionModel.from_pretrained(model_path + "/checkpoint-9500/unet")

vae = AutoencoderKL.from_pretrained(model_path + "/vae")
                   

tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")

feature_ex = CLIPImageProcessor.from_pretrained(model_path + "/feature_extractor")
noise_scheduler = PNDMScheduler.from_pretrained(model_path + "/scheduler")

pipe = StableDiffusionPipeline.from_pretrained(model_path, revision="fp16", safety_checker=None, torch_dtype=torch.float16, ).to("cuda:0")


def aggregate(df,labels, byGender):
    """
    The function `aggregate` takes a dataframe, a list of column labels, and a boolean flag indicating
    whether to aggregate by gender, and returns the aggregated values based on the specified columns and
    grouping.
    
    :param df: The input dataframe that you want to aggregate
    :param labels: The `labels` parameter is a list of column names that you want to aggregate. These
    column names should exist in the DataFrame `df`
    :param byGender: The parameter "byGender" is a boolean value that determines whether the aggregation
    should be performed by gender or not. If "byGender" is set to True, the aggregation will be done
    separately for each gender. If it is set to False, the aggregation will be done for the entire
    dataset without
    :return: the aggregated data based on the specified labels and whether to aggregate by gender or
    not.
    """
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
    """
    The function "draw_x_times" takes a list and an integer as input, and returns a new list with x
    randomly chosen elements from the input list.
    This function is called if a specific amount of genders or labels are to draw randomly
    
    :param list: A list of elements from which to randomly choose
    :param x: The number of times you want to draw an element from the list
    :return: a list of x elements randomly chosen from the input list.
    """
    result_list = [random.choice(list) for _ in range(x)]
    return result_list


label_subset = ['Edema','Cardiomegaly','Support Devices','Atelectasis','Pleural Effusion','Lung Opacity'] 
gender_tokens = ['female', 'male']

def determine_num_prompts(gender_switch,label_switch, num_samples,label_subset,gender_tokens, data_subset):
    """
    The function determines the number of prompts, random genders, and random labels based on the given
    parameters.
    
    :param gender_switch: The `gender_switch` parameter determines whether or not to include gender as a
    factor in determining the number of prompts. If `gender_switch` is `True`, gender will be
    considered. If `gender_switch` is `False`, gender will not be considered
    :param label_switch: The `label_switch` parameter determines whether or not to switch the labels. If
    `label_switch` is `True`, the labels will be switched. If `label_switch` is `False`, the labels will
    not be switched
    :param num_samples: The number of samples you want to generate
    :param label_subset: The `label_subset` parameter is a list of labels that you want to include in
    the random selection
    :param gender_tokens: The `gender_tokens` parameter is a list of tokens representing different
    genders. It could be something like `['female', 'male']` or `['F', 'M']`
    :param data_subset: The `data_subset` parameter is a subset of data that contains information about
    different samples or instances. It could be a dataframe or any other data structure that holds the
    relevant information for each sample
    :return: three values: num_samples, random_genders, and random_labels.
    """
    if label_switch and not gender_switch:
        label_subset = ['Edema','Cardiomegaly','Lung Opacity','Atelectasis']

    if not label_switch and not gender_switch:
        label_aggregation = aggregate(data_subset,label_subset,False)
        limit = num_samples
        num_samples = 0
        random_labels = [] 
        for label_index in label_subset:
            label_left = label_aggregation[label_index]
            num_samples_dis = int(limit - label_left )
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
    """
    The function `generate_random_string` generates a specified number of random strings consisting of
    lowercase letters and digits, with dashes inserted after every 8 characters.
    These resemble the random DICOM file names in MIMIC-CXR.
    
    :param num_samples: The parameter `num_samples` represents the number of random strings you want to
    generate
    :return: The function `generate_random_string` returns a list of randomly generated strings. Each
    string consists of groups of 8 characters separated by dashes. The number of strings in the list is
    determined by the `num_samples` parameter.
    """
    random_strings = []
    for i in range(num_samples):

        chars_per_group = 8
        num_groups = 5

        random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=chars_per_group*num_groups))

        # Insert dashes after every 8 characters (except the last group)
        random_string_with_dashes = '-'.join([random_string[i:i+chars_per_group] for i in range(0, chars_per_group*num_groups, chars_per_group)])
        random_strings.append(random_string_with_dashes)

    return random_strings

def store_img(folder,num_samples,gender_list, label_list, dicom_ids, cwd_img):
    """
    The function `store_img` takes in a folder name, number of samples, gender list, label list, dicom
    IDs, and current working directory for images, and saves images in the specified directory with the
    given file names.
    We follow the structure of the file tree in MIMIC-CXR
    
    :param folder: The folder parameter is the name or identifier of the folder where the images will be
    stored
    :param num_samples: The number of samples or images to be stored
    :param gender_list: The `gender_list` parameter is a list that contains the gender information for
    each sample. It is used to generate the prompt for each image
    :param label_list: The `label_list` parameter is a list that contains the labels for each image. It
    is used to create the prompt for generating the image
    :param dicom_ids: The `dicom_ids` parameter is a list of unique identifiers for each DICOM image.
    These identifiers are used to name the saved image files
    :param cwd_img: The parameter `cwd_img` is the current working directory where the images will be
    stored
    """
    cwd = cwd_img
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
    """
    The function `store_sample` takes in various inputs such as folder, number of samples, dicom ids,
    gender list, label list, and label subset, and returns a combined dataframe with the provided
    inputs.
    
    :param folder: The folder parameter is the name of the folder where the samples will be stored
    :param num_samples: The number of samples to generate in the dataframe
    :param dicom_ids: A list of DICOM IDs for each sample. DICOM IDs are unique identifiers for medical
    images
    :param gender_list: A list of genders for each sample
    :param label_list: The `label_list` parameter is a list of labels associated with each sample
    :param label_subset: The `label_subset` parameter is a list of labels that you want to include in
    the final dataframe. It is used to filter the labels and include only those that are present in the
    `label_subset` list
    :return: a combined dataframe that includes the subject_id, study_id, gender, dicom_id, split, and
    one-hot encoded labels.
    """
    
    folder_list = [folder] * num_samples
    df = pd.DataFrame({'subject_id': folder_list, 'study_id': folder_list})
    
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



folder = '22030000' 
num_samples = int(folder[-5:])
data_subset = pd.read_csv(<path to csv>)
training_data = data_subset.loc[data_subset["split"] == "train"]
num_samples, random_genders, random_labels = determine_num_prompts(gender_switch = False,label_switch = False, num_samples = num_samples, label_subset=label_subset,gender_tokens=gender_tokens, data_subset=training_data)
x_ray_files = generate_random_string(num_samples)
output_df = store_sample(folder=folder,num_samples=num_samples,dicom_ids=x_ray_files, gender_list=random_genders,label_list=random_labels, label_subset=label_subset)



if not output_df.columns.equals(data_subset.columns):
    print("Warning: The column order does not match between output_df and data_subset.")

# Append rows of output_df to data_subset
combined_df = pd.concat([data_subset, output_df], ignore_index=True, verify_integrity=True)

# If you want to overwrite data_subset with the combined DataFrame:
# data_subset = combined_df

# If you want to save the combined DataFrame to a new CSV file:
cwd = <path to storage location of new CSV>
path = os.path.join(cwd,folder)  + ".csv"
combined_df.to_csv(path, index=False)

cwd_img = <path to image storage location>

store_img(folder=folder, num_samples=num_samples,gender_list=random_genders,label_list=random_labels,dicom_ids=x_ray_files, cwd_img = cwd_img)
