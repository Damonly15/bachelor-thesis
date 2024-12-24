import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import torch.nn.functional as F
import torchvision
from types import MethodType

class CnnLN(nn.Module): 
     def __init__(self, c, num_classes) -> None:
          super().__init__()
          self.features = nn.Sequential(
            # Layer 0
            nn.Conv2d(3, c, kernel_size=3, stride=1,
                    padding=1, bias=True),
            CustomGroupNorm(c),
            nn.ReLU(),
            # Layer 1
            nn.Conv2d(c, c*2, kernel_size=3,
                    stride=1, padding=1, bias=True),
            CustomGroupNorm(c*2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Layer 2
            nn.Conv2d(c*2, c*4, kernel_size=3,
                    stride=1, padding=1, bias=True),
            CustomGroupNorm(c*4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Layer 3
            nn.Conv2d(c*4, c*8, kernel_size=3,
                    stride=1, padding=1, bias=True),
            CustomGroupNorm(c*8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten()
        )
          self.head = nn.Linear(c*8, num_classes, bias=True)
    
     def get_features(self, x):
        return self.features(x)
     
     def forward_head(self, phi):
       """Forward through the head only."""
       output = self.head(phi)
       return output
     def forward(self, x):             
        output = self.get_features(x)
        output = self.head(output)
        return output
        
class CnnBN(nn.Module): 
     def __init__(self, c, num_classes) -> None:
          super().__init__()
          self.features = nn.Sequential(
            # Layer 0
            nn.Conv2d(3, c, kernel_size=3, stride=1,
                    padding=1, bias=True),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            # Layer 1
            nn.Conv2d(c, c*2, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.BatchNorm2d(c*2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Layer 2
            nn.Conv2d(c*2, c*4, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.BatchNorm2d(c*4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Layer 3
            nn.Conv2d(c*4, c*8, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.BatchNorm2d(c*8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten()
        )
          self.head = nn.Linear(c*8, num_classes, bias=True)
    
     def get_features(self, x):
        return self.features(x)
     
     def forward_head(self, phi):
       """Forward through the head only."""
       output = self.head(phi)
       return output
     def forward(self, x):             
        output = self.get_features(x)
        output = self.head(output)
        return output
   

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

class CustomGroupNorm(nn.GroupNorm):
    def __init__(self, num_channels):
        # Initialize with num_groups = 1 (in that case group norm is equvalent to layer norm) and provided num_channels
        super(CustomGroupNorm, self).__init__(num_groups=1, num_channels=num_channels)

