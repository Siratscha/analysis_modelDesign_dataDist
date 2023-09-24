from transformers import CLIPTokenizer
import os
import pandas as pd
import re
import ast


tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")



def extract_impression(gender, file_path):
    if gender == "M":
        gender = "male -"
    else:
        gender = "female -"
    with open(file_path, 'r') as file:
        content = file.read()
        
        # Extract the Impression section from the medical report using regular expressions
        matches = re.findall(r'IMPRESSION:(.*?)(?=^$|\Z)', content ,re.MULTILINE | re.DOTALL) 
        
        if matches:
            impressions = [gender + match.replace('\n', '').strip() for match in matches]
        else:
            # Extract the Findings section if Impression section couldn't be found
            matches = re.findall(r'FINDINGS:(.*?)(?=^$|\Z)', content, re.MULTILINE | re.DOTALL)
            impressions = [gender + match.replace('\n', '').strip() for match in matches]
        
        if len(impressions) > 0:
            # Find the index of the "NOTIFICATION" section
            notification_index = impressions[0].find("NOTIFICATION")

            if notification_index != -1:
                # If "NOTIFICATION" section is found, extract the text before it
                impressions = [impressions[0][:notification_index].strip()]

        return impressions

def filter_impressions(impressions):
    MAX_TOKENS = 77
    impressions = ast.literal_eval(impressions)
    if  len(impressions) > 0: 
        original_impression = impressions[0]
        tokenized_impressions = tokenizer.tokenize(original_impression)

        if len(original_impression) >= 7 and len(tokenized_impressions) < MAX_TOKENS:
            return original_impression

    else:
        return []
    


def store_imp():
    samples = pd.read_csv(r"C:\Users\rankl\Documents\uni\Thesis\Development\modelDesign_bias_CXR\diffusers\huggingface_dataset\full_labels_lessNoFinding.csv")
    i = 0
    result_dict = {}
    impression_list = []
    img_paths = []
    limit = 50000
    for i, item in samples.iterrows():
        if i % limit == 0:
            print(i)
        img_dir = "/work/srankl/thesis/modelDesign_bias_CXR/data/MIMICCXR/physionet.org/files/mimic-cxr-jpg/2.0.0/files"
        dir = r"C:\Users\rankl\Documents\uni\Thesis\Development\modelDesign_bias_CXR\data\MIMICCXR\mimic-cxr-reports\files"
        folder, subject_id, study_id = str(int(item["subject_id"]))[:2],  str(int(item["subject_id"])), str(int(item["study_id"]))
        file_path = os.path.join(dir, "p" + folder, "p" + subject_id, "s" + study_id + ".txt")
        img_file_name = str(item["dicom_id"]) + ".jpg"
        image_file_path = os.path.join(img_dir, "p" + folder, "p" + subject_id, "s" + study_id, img_file_name)

        impression_list.append(extract_impression(item["gender"],file_path))
        img_paths.append(image_file_path)


    samples["impressions"] = pd.Series(impression_list[:len(samples)])
    samples["image_paths"] = pd.Series(img_paths[:len(samples)]) 
    return samples

    


#samples = store_imp()
#samples.to_csv(r"C:\Users\rankl\Documents\uni\Thesis\Development\modelDesign_bias_CXR\diffusers\huggingface_dataset\less_NoFindings_fullAP_PA.csv", index = False, sep='|')
samples = pd.read_csv(r"C:\Users\rankl\Documents\uni\Thesis\Development\modelDesign_bias_CXR\diffusers\huggingface_dataset\less_NoFindings_fullAP_PA.csv", delimiter="|")
samples["filtered_impressions"] = samples["impressions"].apply(lambda x: filter_impressions(x))
samples.to_csv(r"C:\Users\rankl\Documents\uni\Thesis\Development\modelDesign_bias_CXR\diffusers\huggingface_dataset\less_NoFindings_fullAP_PA.csv", index = False, sep='|')


#samples = pd.read_csv(r"C:\Users\rankl\Documents\uni\Thesis\Development\modelDesign_bias_CXR\diffusers\huggingface_dataset\total_AP_PA_ds.csv")

#impressions_df = pd.read_csv(r"C:\Users\rankl\Documents\uni\Thesis\Development\modelDesign_bias_CXR\diffusers\huggingface_dataset\less_NoFindings_fullAP_PA.csv", delimiter= "|")
impressions_df = samples
impressions_df.dropna(subset=['image_paths', 'filtered_impressions'], inplace=True)
mask = impressions_df['filtered_impressions'].str.contains(r'\[\]')

# Drop the rows where the column contains the string '[]'
impressions_df = impressions_df[~mask]
#impressions_df.to_csv(r"C:\Users\rankl\Documents\uni\Thesis\Development\modelDesign_bias_CXR\test_df_export.csv", index = False, sep='|')

print(len(impressions_df))
import json

def create_jsonl_splits(impressions_df):
    # Define the splits for creating separate files
    splits = ["train", "test", "validate"]

    # Group the DataFrame by split
    split_groups = impressions_df.groupby("split")

    # Iterate over the splits
    for split in splits:
        # Get the corresponding split group
        split_group = split_groups.get_group(split)
        
        # Create a list to store the examples
        examples = []
        
        # Iterate over the rows in the split group
        for _, row in split_group.iterrows():
            # Create an example dictionary with "image" and "text" fields
            example = {
                "image": row["image_paths"],
                "text": row["filtered_impressions"]
            }
            
            # Append the example to the list
            examples.append(example)
        
        # Create the file path for the split
        file_path = f"{split}.jsonl"
        
        # Write the examples to the .jsonl file
        with open(file_path, "w") as file:
            for example in examples:
                # Write each example as a separate line in the file
                file.write(json.dumps(example) + "\n")

create_jsonl_splits(impressions_df)



