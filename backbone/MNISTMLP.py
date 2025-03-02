# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from backbone import MammothBackbone, num_flat_features, xavier


class MNISTMLP(MammothBackbone):
    """
    Network composed of two hidden layers, each containing 100 ReLU activations.
    Designed for the MNIST dataset.
    """

    def __init__(self, input_size: int, output_size: int, cpt: int=-1) -> None:
        """
        Instantiates the layers of the network.

        Args:
            input_size: the size of the input data
            output_size: the size of the output
        """
        super(MNISTMLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, 100)
        self.fc2 = nn.Linear(100, 100)

        self._features = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
        )
        self.feature_dim=100
        if cpt==-1:
            self.classifier = nn.Linear(100, self.output_size)
        else:
            self.classifier = nn.ModuleList([nn.Linear(100, cpt) for _ in range(self.output_size//cpt)])
        self.net = nn.Sequential(self._features, self.classifier)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self.net.apply(xavier)

    def forward(self, x: torch.Tensor, task_label=None, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, input_size)

        Returns:
            output tensor (output_size)
        """
        x = x.view(-1, num_flat_features(x))


        feats = self._features(x)

        if returnt == 'features':
            return feats

        if task_label is None:
            out = self.classifier(feats)
        elif torch.is_tensor(task_label):
            batch_size = feats.shape[0]
            out = torch.zeros((batch_size, self.classifier[0].out_features), device=feats.device)

            unique_labels = torch.unique(task_label)
            for label_idx in unique_labels:
                mask = (label_idx == task_label)
                feature_head = feats[mask]
                out[mask] = self.classifier[label_idx](feature_head)
        else:
            out = self.classifier[task_label](feats)

        if returnt == 'out':
            return out
        elif returnt in ['both', 'all']:
            return (out, feats)

        raise NotImplementedError("Unknown return type")

    def to(self, device):
        super().to(device)
        self.device = device
        return self
