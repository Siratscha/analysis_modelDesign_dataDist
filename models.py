import torch.nn as nn
from torchvision import models , transforms as transforms
import torch

import torch.nn.functional as nnf


class DenseNet121(nn.Module):
    def __init__(self, N_LABELS):
        super(DenseNet121, self).__init__()
        self.densenet = models.densenet121(weights='DEFAULT')
        self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        

        #for param in self.densenet.parameters():
        #    param.requires_grad = False
        
        #layers_to_finetune = ['denseblock4', 'transition3'] #'denseblock3', 'transition2',

        #for name, param in self.densenet.named_parameters():
        #    if any(layer in name for layer in layers_to_finetune):
        #        param.requires_grad = True
        
        num_ftrs = self.densenet.classifier.in_features
        dropout_prob = 0.2
        
        
        #self.densenet.classifier = nn.Sequential(
        #    nn.Linear(num_ftrs,1024),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout(dropout_prob),
        #    nn.Linear(1024, N_LABELS),
        #    nn.Sigmoid())
        self.densenet.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
        #self.densenet.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Softmax(dim=1))
        #self.densenet.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.LogSoftmax(dim=1))



    def forward(self, x):
        x = self.densenet(x)
        return x
