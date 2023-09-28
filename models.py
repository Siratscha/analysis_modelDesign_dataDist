# defines the DenseNet121

import torch.nn as nn
from torchvision import models , transforms as transforms

class DenseNet121(nn.Module):
    def __init__(self, N_LABELS):
        super(DenseNet121, self).__init__()
        self.densenet = models.densenet121(weights='DEFAULT')
        # include conv2d layer to take only one channel
        self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        
        num_ftrs = self.densenet.classifier.in_features

        

        self.densenet.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())


    def forward(self, x):
        x = self.densenet(x)
        return x
