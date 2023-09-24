import torch
import torch.nn as nn
import random


from torchvision import transforms 

from data.MIMICCXR.mimic_cxr_dataset import MIMICCXR
from models import DenseNet121
from transformers import ViTForImageClassification

import numpy as np
import pandas as pd

from tqdm import tqdm
import os


from sklearn.metrics import  roc_auc_score



def train(input_csv,  PATH_TO_IMAGES, classes, epochs,criterion,batch_size,model_name, modeltype = 'densenet', device = 'Optional', 
        lr = 0.0001,image_size=256, mode = 'new'):

    normalize = transforms.Normalize(mean=[0.485],
                                     std=[0.229])
    

    merged_df = pd.read_csv(input_csv)

    train_df = merged_df[merged_df['split'] == 'train']
    #train_df = train_df.sample(frac=0.05)
    val_df = merged_df[merged_df['split'] == 'validate']
    


    # Training parameters
    BATCH_SIZE = batch_size

    num_classes = len(classes)

    WORKERS = 10  # mean: how many subprocesses to use for data loading.
    num_epochs = epochs  # number of epochs to train for (if early stopping is not triggered)

    random_seed = random.randint(0,100)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)  

    train_dataset = MIMICCXR(train_df, PATH_TO_IMAGES, classes, model_type=modeltype)
    train_dataset.transform = transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(15),
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.Grayscale(num_output_channels=1), #transform the sd generated images to grayscale
                                    transforms.ToTensor(),
                                    normalize
                                ])

    val_dataset = MIMICCXR(val_df, PATH_TO_IMAGES, classes, model_type=modeltype)
    val_dataset.transform = transforms.Compose([ 
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    normalize
                                ])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
        shuffle=True,num_workers=WORKERS, pin_memory=True) #num_workers=WORKERS,

    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=BATCH_SIZE,
         shuffle=True,num_workers=WORKERS,  pin_memory=True)         

    if modeltype == 'densenet':
        model = DenseNet121(num_classes)
        
    if modeltype == 'vit':
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-384", num_channels=1,num_labels=num_classes, ignore_mismatched_sizes=True )


    if mode == 'new': 
        optimizer =  torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5, amsgrad=True)
        epoch = 0
        best_loss = float('inf')
        best_auc = 0
        train_losses = torch.tensor([])
        val_losses = torch.tensor([])
        val_auc_scores = torch.tensor([])

    
    if mode == 'resume':#and modeltype == 'densenet':
        file_name_model = str(model_name) + '.pth'
        
        subfolder = 'outputs/weights'
        subfolder = os.path.join(subfolder,file_name_model)
        checkpoint = torch.load(subfolder)

        model = checkpoint['model']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = checkpoint['optimizer']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #epoch = checkpoint['epoch']

        file_name_stats = str(model_name) + '_stats.pth'
        subfolder_stats = 'outputs'
        subfolder_stats = os.path.join(subfolder_stats,modeltype,file_name_stats)
        checkpoint_stats = torch.load(subfolder_stats)

        epoch = checkpoint_stats['epoch']
        best_epoch = checkpoint_stats['best_epoch']
        best_auc = checkpoint_stats['best_auc']
        best_loss = checkpoint_stats['best_loss']
        train_losses = checkpoint_stats['train_losses']
        val_losses = checkpoint_stats['val_losses']
        val_auc_scores = checkpoint_stats['val_aucs']
        

    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    model = model.to(device)


    if criterion == 'BCE':
        criterion = nn.BCELoss().to(device)
    elif criterion == 'CE':
        criterion = nn.CrossEntropyLoss().to(device)
    elif criterion == 'BCELL':
        criterion = nn.BCEWithLogitsLoss().to(device)
    

    for epoch_iter in tqdm(range(epoch, num_epochs)): 
        
        epoch_loss_train = train_epoch(
                               model=model,
                               device=device,
                               optimizer=optimizer,
                               train_loader=train_loader,
                               criterion=criterion)
        
        epoch_loss_val = 0.0
        train_losses = torch.cat([train_losses, torch.tensor([epoch_loss_train])])

        val_auc, epoch_loss_val, *rest = valid_epoch(model=model,
                                        device=device, 
                                        val_loader=val_loader, 
                                        criterion=criterion, 
                                        num_classes=num_classes)
  
        val_losses = torch.cat([val_losses, torch.tensor([epoch_loss_val])])
        val_auc_scores = torch.cat([val_auc_scores, torch.tensor([val_auc])])

        print('\nEpoch {}: Training Loss = {:.7f}, Validation Loss = {:.7f}, Validation AUC = {:.4f}'.format(epoch_iter+1, epoch_loss_train, epoch_loss_val, val_auc))

        if val_auc > best_auc:
            best_auc = val_auc
    
        if epoch_loss_val < best_loss :
            best_loss = epoch_loss_val
            best_epoch = epoch

            checkpoint = {
                'model': model,
                'optimizer':optimizer,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }
            file_name = str(model_name) + '.pth'
            subfolder = 'outputs/weights'
            subfolder = os.path.join(subfolder,file_name)
            torch.save(checkpoint, subfolder)
        
        if (((epoch_iter +1 ) - best_epoch) >= 3):
            new_learning_rate = optimizer.param_groups[0]['lr'] / 2
            optimizer.param_groups[0]['lr'] = new_learning_rate

        checkpoint_stats = {
                    'epoch': epoch_iter + 1,
                    'best_loss': best_loss,
                    'best_epoch': best_epoch,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_aucs':val_auc_scores,
                    'best_auc':best_auc
            }
        
        file_name = str(model_name) + '_stats.pth'
        subfolder = 'outputs'
        subfolder = os.path.join(subfolder,modeltype,file_name)
        torch.save(checkpoint_stats, subfolder)

    print('Finished Training')

def train_epoch(device, train_loader, optimizer, criterion, model):
    model.train()
    avg_losses = []
    for batch_idx, (images, labels) in enumerate(train_loader):
            
            
            images = images.to(device)
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            

            # forward + backward + optimize
            outputs = model(images)
            
            loss = torch.zeros(1).to(device).float()
            for task in range(labels.shape[1]):
                task_output = outputs.logits[:,task] #outputs[:,task] outputs.logits
                task_label = labels[:,task]
                mask = ~torch.isnan(task_label)
                task_output = task_output[mask]
                task_label = task_label[mask]
                if len(task_label) > 0:
                    task_loss = criterion(task_output.float(), task_label.float())
                    loss += task_loss
            loss = loss.sum()
            #criterion(outputs, labels)
            loss.backward()

            avg_losses.append(loss.detach().cpu().numpy())

            optimizer.step()  # update weights

    return np.mean(avg_losses)    
           

def valid_epoch(val_loader, device, model, criterion, num_classes):
    model.eval()
    avg_loss = []
    test_pred_list={} 
    test_true_list={}
    for task in range(num_classes):
        test_pred_list[task] = []
        test_true_list[task] = []

    for batch_idx, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)
        BATCH_SIZE = images.shape[0]
            
        with torch.no_grad():
            outputs = model(images)
            loss = torch.zeros(1).to(device).double()
            for task in range(labels.shape[1]):
                task_output = outputs.logits[:,task] #outputs[:,task]
                task_label = labels[:,task]
                mask = ~torch.isnan(task_label)
                task_output = task_output[mask]
                task_label = task_label[mask]
                if len(task_label) > 0:
                    task_loss = criterion(task_output.float(), task_label.float())
                    loss += task_loss
                test_pred_list[task].append(task_output.detach().cpu().numpy())
                test_true_list[task].append(task_label.detach().cpu().numpy())
            loss = loss.sum()
            avg_loss.append(loss.detach().cpu().numpy())
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


    return auc, np.mean(avg_loss), task_aucs, test_pred_list, test_true_list