import os
import numpy as np
import torch

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from backbone.MNISTMLP import MNISTMLP
from datasets.seq_mnist import SequentialMNIST
from datasets.perm_mnist import PermutedMNIST
from backbone.ResNet18LayerNorm import resnet18layernorm
from datasets.seq_cifar10 import SequentialCIFAR10
from datasets.seq_tinyimagenet import SequentialTinyImagenet


def adjust_outputs(outputs, class_labels, classes_per_task):
    indices = classes_per_task * class_labels.unsqueeze(1) + torch.tensor([i for i in range(classes_per_task)], dtype=torch.int64, device=outputs.device)
    new_outputs = outputs.gather(1, indices)
    return new_outputs

def get_pretrained(args):
    if args.dataset == "seq-mnist":
        pretrained_model = MNISTMLP(np.prod(SequentialMNIST.SIZE), SequentialMNIST.N_CLASSES)
        pretrained_model.load_state_dict(torch.load(mammoth_path + "/pretrained_models/seq_mnist_20epochs.pth"))
    elif args.dataset == "seq-cifar10":
        pretrained_model = resnet18layernorm(nclasses = SequentialCIFAR10.N_CLASSES, inputs_size = SequentialCIFAR10.SIZE[0])
        pretrained_model.load_state_dict(torch.load(mammoth_path + "/pretrained_models/seq_cifar10_100epochs.pth"))
    elif args.dataset == "perm-mnist":
        pretrained_model = MNISTMLP(np.prod(PermutedMNIST.SIZE), PermutedMNIST.N_CLASSES)
        pretrained_model.load_state_dict(torch.load(mammoth_path + "/pretrained_models/perm_mnist_12epochs.pth"))
    elif args.dataset == "seq-tinyimg":
        pretrained_model = resnet18layernorm(nclasses = SequentialTinyImagenet.N_CLASSES, inputs_size = SequentialTinyImagenet.SIZE[0])
        pretrained_model.load_state_dict(torch.load(mammoth_path + "/pretrained_models/seq_tinyimg_150epochs.pth"))   

    return pretrained_model

