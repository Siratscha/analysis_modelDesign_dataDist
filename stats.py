# this script is called from the main.py script after training the model to calculate the AUC-ROC values


import torch
import matplotlib.pyplot as plt

from torchvision import transforms

from data.MIMICCXR.mimic_cxr_dataset import MIMICCXR
from models import DenseNet121

from torchmetrics.classification import MultilabelAUROC
from sklearn.metrics import roc_auc_score

import numpy as np

import pandas as pd

from tqdm import tqdm


# 
def print_loss_curve(path):
    """
    The function `print_loss_curve` plots the loss curves and AUC-ROC curve for a given path to a
    checkpoint file.
    
    :param path: The `path` parameter is the file path to the checkpoint file that contains the loss and
    AUC ROC values for training and validation
    """
    checkpoint = torch.load(path)
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]
    val_aucs = checkpoint['val_aucs']
    

    num_epochs = val_losses.shape[0]
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax1 = plt.subplots()
    
    # Plotting loss curves
    color = '0.1'  
    ax1.set_xlabel('Epoch', fontsize=24)  
    ax1.set_ylabel('Loss', color=color, fontsize=24)  
    ax1.plot(range(num_epochs), val_losses.detach().numpy(), label='Val. loss', color=color)
    ax1.plot(range(num_epochs), train_losses.detach().numpy(), label='Train loss', color='0.6')  

    ax1.tick_params(axis='both', labelcolor=color, labelsize=20)


    ax2 = ax1.twinx()

    ax2.set_ylabel('AUC ROC', color=color, fontsize=24)  
    ax2.plot(range(num_epochs), val_aucs.detach().numpy(), label='Val. AUC-ROC', color='0.3', ls='--')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=20)


    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    ax1.legend(loc='upper right', bbox_to_anchor=(0.5, 1.15),
          ncol=2, fancybox=True)
    
    ax2.legend(loc='upper left', bbox_to_anchor=(0.5, 1.15),
          ncol=1, fancybox=True)

    # Legend and title with adjusted font size
    fig.tight_layout()


    # Display the plot
    plt.show()



# test loop from https://github.com/mlmed/torchxrayvision
def test_predictions(path2model, input_csv,  PATH_TO_IMAGES, classes, device, gender,modeltype, image_size = 256):
    """
    The function `test_predictions` takes in a path to a trained model, an input CSV file, a path to a
    directory containing images, a list of classes, a device (CPU or GPU), a gender, a model type, and
    an image size. It then loads the test dataset, loads the trained model, performs inference on the
    test dataset, calculates the AUC scores for each task, and returns the AUC scores and the gender.
    
    :param path2model: The path to the saved model file
    :param input_csv: The path to the CSV file containing the dataset information, including the split
    (train, test, etc.) and gender information
    :param PATH_TO_IMAGES: The `PATH_TO_IMAGES` parameter is the path to the directory where the images
    are stored. It is used to load the images for prediction
    :param classes: The `classes` parameter is a list of class labels for the classification task. It
    represents the different categories or classes that the model is trained to predict
    :param device: The "device" parameter specifies whether to use a GPU or CPU for running the model.
    It can take values like "cuda" or "cpu"
    :param gender: The "gender" parameter in the function "test_predictions" is used to specify the
    gender for which the predictions are being made. It is a string that can take two values: "male" or
    "female". This parameter is used to filter the data from the input CSV file and select only the
    :param modeltype: The `modeltype` parameter is not used in the given code. It is not clear what it
    represents or how it is used in the function
    :param image_size: The `image_size` parameter is the size to which the input images will be resized
    before being fed into the model. It is set to 256 pixels in the given code, defaults to 256
    (optional)
    :return: The function `test_predictions` returns two values: `task_aucs` and `gender`.
    """
    normalize = transforms.Normalize(mean=[0.485],
                                     std=[0.229])
    
    merged_df = pd.read_csv(input_csv)

    
    merged_df = merged_df[(merged_df["split"] == "test") & (merged_df["gender"] == gender)]

    test_df = merged_df.sample(frac=1)
    print(len(test_df),", Classes: ", classes)



    test_dataset = MIMICCXR(test_df, PATH_TO_IMAGES, classes=classes)
    test_dataset.transform = transforms.Compose([ 
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    normalize
                                ])
    
    num_classes = len(classes)
    
    BATCH_SIZE = 48
    WORKERS = 10
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,num_workers=WORKERS, shuffle=True, pin_memory=True) 
    file_name = str(path2model) + '.pth' 
    checkpoint = torch.load(file_name)
    # torch.load(path2model, map_location=torch.device('cpu'))
    #
    
    model = checkpoint['model']
    #model = DenseNet121(14)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)

    
    model.eval()
    
    test_pred_list={} 
    test_true_list={}
    for task in range(num_classes):
        test_pred_list[task] = []
        test_true_list[task] = []

    for batch_idx, (images, labels) in tqdm(enumerate(test_loader)):
        images = images.to(device)
        labels = labels.to(device)
        BATCH_SIZE = images.shape[0]
            
        with torch.no_grad():
            outputs = model(images)
            
            for task in range(labels.shape[1]):
                task_output = outputs.logits[:,task]
                task_label = labels[:,task]
                mask = ~torch.isnan(task_label)
                task_output = task_output[mask]
                task_label = task_label[mask]

                test_pred_list[task].append(task_output.detach().cpu().numpy())
                test_true_list[task].append(task_label.detach().cpu().numpy())

    for task in range(len(test_true_list)):
        test_pred_list[task] = np.concatenate(test_pred_list[task])
        test_true_list[task] = np.concatenate(test_true_list[task])
    task_aucs = []
    for task in range(len(test_true_list)):
        if len(np.unique(test_true_list[task]))> 1:
            task_auc = roc_auc_score(test_true_list[task], test_pred_list[task])
            #print(task, task_auc)
            task_aucs.append(task_auc)
        else:
            task_aucs.append(np.nan)
    task_aucs = np.asarray(task_aucs)
    auc = np.mean(task_aucs[~np.isnan(task_aucs)])
    return task_aucs, gender

#path = r"C:\Users\rankl\Documents\uni\Thesis\Development\modelDesign_bias_CXR\outputs\vit\vit_data_subset_0001_stats.pth"
#print_loss_curve(path)
#print_aucroc(path)