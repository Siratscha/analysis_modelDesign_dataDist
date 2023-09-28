# trains an image classifier: either densenet or vit
# outputs the standard deviation and the test results
# parameters are to be defined in "model_configuration.jsonc"

from train import train
from stats import test_predictions, print_loss_curve


import json
import numpy as np

import csv
import os


json_file_path = "model_configuration.jsonc"

with open(json_file_path,"r") as json_file:
    configuration = json.load(json_file)

model = configuration["model"]["architecture"]
criterion = configuration["model"]["criterion"]
lr = configuration["training"]["learning_rate"]
epochs_conf = configuration["training"]["epochs"]
batch_size = configuration["training"]["batch_size"]
image_path = configuration["data"]["image_path"]
labels = configuration["data"]["label_path"]
classes_conf = configuration["data"]["classes"]
cuda_device = configuration["cuda_device"]
model_name = configuration["model"]["name"]
image_size = configuration["model"]["image_size"]


# %%
def main():
    """
    The main function calculates the mean and standard deviation of the area under the ROC curve (AUC)
    for a given model and dataset.
    """
    
    mean_aucs =  np.zeros((2, len(classes_conf)))
    std_aucs = np.zeros((2, len(classes_conf)))
    aucs_scores = [[],[]]
    genders = ['F', 'M']

    subfolder_weights = 'outputs/weights'
    subfolder_weights = os.path.join(subfolder_weights,model_name)


    for i in range(2):
        train(labels, image_path, classes=classes_conf, epochs=epochs_conf,batch_size=batch_size, 
            mode= "new", device= cuda_device, lr=lr,criterion=criterion, modeltype=model, model_name =model_name, image_size=image_size)
        for j,gender in enumerate(genders):
            task_aucs, gender= test_predictions(subfolder_weights, labels, image_path, classes=classes_conf,device=cuda_device,gender=gender,modeltype=model, image_size=image_size)
            aucs_scores[j].append(task_aucs)
            mean_aucs[j] += task_aucs
            
    mean_aucs = mean_aucs/(i + 1)

    # Compute the standard deviation
    for i in range(len(aucs_scores)):
        mean_auroc = np.array(mean_aucs[i])

        squared_diff = (aucs_scores[i] - mean_auroc) ** 2  

        variance = np.mean(squared_diff, axis=0)  

        standard_deviation = np.sqrt(variance)
        std_aucs[i] += standard_deviation

        
    print("Standard Deviation:", std_aucs)


    subfolder = 'outputs'
    subfolder = os.path.join(subfolder,model)
    
    os.makedirs(subfolder, exist_ok=True) 
    column_names = classes_conf
    filename = model_name + '.csv'
    filepath = os.path.join(subfolder,filename)

    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(column_names)
        
        writer.writerows(mean_aucs)
        writer.writerows(std_aucs)

    print(mean_aucs)
    


if __name__ == "__main__":
    #main() #<- uncomment to train a model
    path = <path to trained model>
    
    #print_loss_curve(path) #<- uncomment to see the statistics of a model 

    

