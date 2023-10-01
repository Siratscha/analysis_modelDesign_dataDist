# utils script to extract the impression section in the medical reports
# 

from transformers import CLIPTokenizer
import os
import pandas as pd
import re
import ast
import json


tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")



def extract_impression(gender, file_path):
    """
    The function `extract_impression` takes a gender and a file path as input, reads the content of the
    file, extracts the Impression section from the medical report using regular expressions, and returns
    a list of impressions with the gender prefix.
    
    :param gender: The gender of the patient. It can be either "M" for male or "F" for female
    :param file_path: The file path is the location of the medical report file that you want to extract
    the impression from
    :return: The function `extract_impression` returns a list of impressions extracted from a medical
    report. Each impression is prefixed with the gender (either "male -" or "female -").
    """
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
        
        # sometimes the Header "Impression" is missing. Then we use a workaround and extract the text before "Notification" section which refers to "Impression" again
        if len(impressions) > 0:
            # Find the index of the "NOTIFICATION" section
            notification_index = impressions[0].find("NOTIFICATION")

            if notification_index != -1:
                # If "NOTIFICATION" section is found, extract the text before it
                impressions = [impressions[0][:notification_index].strip()]

        return impressions

def filter_impressions(impressions):
    """
    The function `filter_impressions` takes a list of impressions, tokenizes the first impression, and
    returns the original impression if it has a length of at least 7 characters and the tokenized
    impression has fewer than 77 tokens. Otherwise, it returns an empty list.
    
    :param impressions: The parameter "impressions" is expected to be a list of strings
    :return: The function `filter_impressions` returns either the original impression or an empty list,
    depending on certain conditions.
    """
    # max tokens refers to the limit of the CLIP tokenizer. in this case it's 77
    MAX_TOKENS = 77
    impressions = ast.literal_eval(impressions)
    if  len(impressions) > 0: 
        original_impression = impressions[0]
        tokenized_impressions = tokenizer.tokenize(original_impression)

        if len(original_impression) >= 7 and len(tokenized_impressions) < MAX_TOKENS:
            return original_impression

    else:
        return []
    


samples = pd.read_csv(<path to csv containing the labels and image paths>, delimiter="|")
samples["filtered_impressions"] = samples["impressions"].apply(lambda x: filter_impressions(x))
samples.to_csv(<path to store filtered csv>, index = False, sep='|')



impressions_df = samples
impressions_df.dropna(subset=['image_paths', 'filtered_impressions'], inplace=True)
mask = impressions_df['filtered_impressions'].str.contains(r'\[\]')

# Drop the rows where the column contains the string '[]'
impressions_df = impressions_df[~mask]

#print(len(impressions_df))



def create_jsonl_splits(impressions_df):
    """
    The function `create_jsonl_splits` takes an impressions dataframe and splits it into separate train,
    test, and validate groups, then creates separate JSONL files for each group.
    
    :param impressions_df: impressions_df is a pandas DataFrame that contains the data for creating the
    JSONL splits. It should have the following columns:
    """
    # Define the splits for creating separate files
    splits = ["train", "test", "validate"]

    split_groups = impressions_df.groupby("split")

    for split in splits:
        
        split_group = split_groups.get_group(split)
        
        examples = []
        
       
        for _, row in split_group.iterrows():
            # Create a dictionary with "image" and "text" fields
            example = {
                "image": row["image_paths"],
                "text": row["filtered_impressions"]
            }
            
            
            examples.append(example)
        
        
        file_path = f"{split}.jsonl"
        
        # Write the samples to the .jsonl file
        with open(file_path, "w") as file:
            for example in examples:
                # Write each sample as a separate line in the file
                file.write(json.dumps(example) + "\n")

create_jsonl_splits(impressions_df)



