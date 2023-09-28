# analysis_modelDesign_dataDist
Submission repository of my master thesis called "Uncovering the Effect of Model Design 
and Data Distribution on Bias: An Analysis".

This repository contains the python implementation of the image classification, image 
generation, the metric imbalance score and the data sampling algorithms.

Furthermore this repository contains the detailed settings and results of the conducted experiments.

# Folders

data/MIMICCXR: contains the data for training the image classifier. We cleaned all images and clinical data 
from this repository for data protection reasons. 
Only credentialed users who sign the data usage aggreements can access the files of MIMIC-CXR (https://physionet.org/content/mimic-cxr/2.0.0/). 

diffusers: contains the scripts for fine tuning the stable diffusion pipeline including utils regarding the huggingface dataset,
the image generation during inference, and examples

experiments: contains the scripts for creating the scenarios used for studies 1 and 2, the results of the experiments including calculations

images: contains figures used in the text of the thesis

# Files
main.py, model_configuration.jsonc, models.py, stats.py, train.py are used for the image classification


 

