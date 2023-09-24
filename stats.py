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



def print_loss_curve(path):
    checkpoint = torch.load(path)
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]
    val_aucs = checkpoint['val_aucs']
    

    num_epochs = val_losses.shape[0]
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax1 = plt.subplots()
    
    # Plotting loss curves
    color = '0.1'  # Grayscale color
    ax1.set_xlabel('Epoch', fontsize=24)  # Adjust font size
    ax1.set_ylabel('Loss', color=color, fontsize=24)  # Adjust font size
    ax1.plot(range(num_epochs), val_losses.detach().numpy(), label='Val. loss', color=color)
    ax1.plot(range(num_epochs), train_losses.detach().numpy(), label='Train loss', color='0.6')  # Lighter grayscale
    #ax1.tick_params(axis='y', labelcolor=color, labelsize=16)
    ax1.tick_params(axis='both', labelcolor=color, labelsize=20)

    # Plotting AUROC curves
    ax2 = ax1.twinx()

    ax2.set_ylabel('AUC ROC', color=color, fontsize=24)  # Adjust font size
    ax2.plot(range(num_epochs), val_aucs.detach().numpy(), label='Val. AUC-ROC', color='0.3', ls='--')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=20)

    # Shrink current axis's height by 10% on the bottom
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    ax1.legend(loc='upper right', bbox_to_anchor=(0.5, 1.15),
          ncol=2, fancybox=True)
    
    ax2.legend(loc='upper left', bbox_to_anchor=(0.5, 1.15),
          ncol=1, fancybox=True)

    # Legend and title with adjusted font size
    fig.tight_layout()
    #fig.legend(loc='lower center', fontsize=20)  # Adjust font size
    #plt.title('Loss and AUROC Curves', fontsize=20)  # Adjust font size

    # Display the plot
    plt.show()



#print_loss_curve()

def test_predictions(path2model, input_csv,  PATH_TO_IMAGES, classes, device, gender,modeltype, image_size = 256):
    normalize = transforms.Normalize(mean=[0.485],
                                     std=[0.229])
    
    merged_df = pd.read_csv(input_csv)

    
    merged_df = merged_df[(merged_df["split"] == "test") & (merged_df["gender"] == gender)]

    test_df = merged_df.sample(frac=1)
    print(len(test_df),", Classes: ", classes)



    test_dataset = MIMICCXR(test_df, PATH_TO_IMAGES, classes=classes, model_type=modeltype)
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