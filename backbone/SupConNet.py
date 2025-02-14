""""
Extension of any backbone for contrastive loss in a detached feature space. 
Adapted from https://github.com/chaht01/Co2L/blob/main/networks/resnet_big.py#L164 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import MammothBackbone

class SupConWrapper(MammothBackbone):
    """backbone + projection head"""
    def __init__(self, encoder:MammothBackbone, head="linear", feat_dim=128):
        super(SupConWrapper, self).__init__()

        self.encoder = encoder
        dim_in = encoder.feature_dim

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def reinit_head(self):
        for layers in self.head.children():
            if hasattr(layers, 'reset_parameters'):
                layers.reset_parameters()

    def forward(self, x, task_label=None, returnt='out', norm=True):
        
        if returnt == 'supcon': # training with contrastive loss
            out, encoded = self.encoder(x, task_label, returnt='both')
        else: 
            return self.encoder.forward(x, task_label, returnt)

        if norm: feat = F.normalize(self.head(encoded), dim=1)
        else: feat = self.head(encoded)
        
        return out, feat
