# torch image dataset adjusted to MIMIC-CXR
import torch
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image




# %%
class MIMICCXR(Dataset):
    def __init__(self, csv_labels, root_dir, classes, transform=None):
        
        """
            Dataset class representing MIMICCXR dataset
            
            Arguments:
            csv_labels: A path to a csv file containing the labels.
            root_dir: Path to the image directory on the server
            transform: Optional if we want to transform the sample images
            
            Returns: image data and label
            
        """
        self.img_labels = csv_labels 
        self.img_dir = root_dir
        self.transform = transform
        self.classes = classes


    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
                    idx = idx.tolist()
        
        item = self.img_labels.iloc[idx]
        folder, subject_id, study_id = "p" + str(int(item["subject_id"]))[:2], "p" + str(int(item["subject_id"])), "s" + str(int(item["study_id"]))
        img_file_name = str(item["dicom_id"]) + ".jpg"
        img_path = os.path.join(self.img_dir,folder, subject_id, study_id, img_file_name)

        image = None
        try:
                image = Image.open(img_path) 
                
        except FileNotFoundError:
            
            print(f"The file '{img_path}' does not exist.")
            return 0,0,""


        if self.transform: 
            image = self.transform(image)
  
       
        label = torch.FloatTensor(np.zeros(len(self.classes), dtype=int))
        for i in range(0, len(self.classes)):
            # use only two labels (0,1) to make it a multi-label case
            if (self.img_labels[self.classes[i].strip()].iloc[idx].astype('float') > 0):
                label[i] = self.img_labels[self.classes[i].strip()].iloc[idx].astype('float')

        return image, label


