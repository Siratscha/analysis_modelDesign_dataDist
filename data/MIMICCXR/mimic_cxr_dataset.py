# torch image dataset adjusted to MIMIC-CXR
import torch
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image




# %%
class MIMICCXR(Dataset):
   # The `__init__` method is the constructor of the `MIMICCXR` class. It initializes the object and
   # sets its initial state. In this case, it takes in several arguments:
    def __init__(self, csv_labels, root_dir, classes, transform=None):
        """
        The function is a constructor for a dataset class representing the MIMICCXR dataset, taking in
        arguments for csv_labels, root_dir, classes, and an optional transform.
        
        :param csv_labels: A path to a csv file containing the labels. This csv file should have two
        columns - one for the image file names and another for the corresponding labels
        :param root_dir: The root directory where the image files are stored. This is the path to the
        directory on the server where the images are located
        :param classes: The "classes" parameter is a list that contains the names of the different
        classes or categories in the dataset. 
        :param transform: The transform parameter is an optional argument that allows you to apply
        transformations to the sample images. Transformations can include resizing, cropping, rotating,
        flipping, normalizing, etc. 
        """
        
        self.img_labels = csv_labels 
        self.img_dir = root_dir
        self.transform = transform
        self.classes = classes


    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        """
        The `__getitem__` function retrieves an image and its corresponding label from a dataset, given
        an index.
        
        :param idx: The `idx` parameter is the index of the item that you want to retrieve from the
        dataset. It is used to access the corresponding image and label data
        :return: The `__getitem__` method returns the image and label corresponding to the given index.
        The image is returned as a PIL image object, and the label is returned as a torch.FloatTensor
        object.
        """
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


