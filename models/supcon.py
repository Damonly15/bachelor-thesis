"""
Supervised Contrastive loss and experience replay. A modification of this method: https://openaccess.thecvf.com/content/ICCV2021/papers/Cha_Co2L_Contrastive_Continual_Learning_ICCV_2021_paper.pdf

Example usage:
    model = 
    loss = 

"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from backbone.SupConNet import SupConWrapper
from models.er_bounds import ErBounds
from models.er_portion import ErPortion
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


import torchvision.transforms as transforms


class SupCon(ErBounds):
    NAME = 'supcon'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual', 'cssl']

    @staticmethod
    def get_parser() -> ArgumentParser:
        """
        Returns an ArgumentParser object with predefined arguments for the Er model.

        Besides the required `add_management_args` and `add_experiment_args`, this model requires the `add_rehearsal_args` to include the buffer-related arguments.
        """
        parser = ArgumentParser(description='Continual learning via SupCon Experience Replay.')
        add_rehearsal_args(parser)
        parser.add_argument('--temperature', type=float, default=0.07, help='temperature for loss function')
        parser.add_argument('--alpha', type=float, default=1.0)
        parser.add_argument('--asym', action='store_true', help='use asymmetric loss')
        parser.add_argument('--contrast_mode', type=str, default="all")
        parser.add_argument('--base_temperature', type=float, default=0.07)
        return parser

    def __init__(self, backbone, loss, args, transform):
        """
        The ER model maintains a buffer of previously seen examples and uses them to augment the current batch during training.
        """
        super(SupCon, self).__init__(backbone, loss, args, transform)
        # transform backbone into a supcon compatible backbone
        self.net = SupConWrapper(backbone, head="linear", feat_dim=128) 
        self.contrast_mode = args.contrast_mode
        self.temperature=args.temperature
        self.base_temperature=args.base_temperature
        self.asym=args.asym
        self.alpha=args.alpha
        self.to_pil = transforms.ToPILImage()


    def contrastive_loss(self, features, labels, batch_size): 
        """"
        Logic for contrastive loss computation
        """
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(self.device)
        mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device), 0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        if self.asym:
            target_labels= list(range(self.current_task*self.dataset.N_CLASSES_PER_TASK, (self.current_task+1)*self.dataset.N_CLASSES_PER_TASK))
            curr_class_mask = torch.zeros_like(labels)
            for tc in target_labels:
                curr_class_mask += (labels == tc)
            curr_class_mask = curr_class_mask.view(-1).to(self.device)
            loss = curr_class_mask * loss.view(anchor_count, batch_size)
        
        loss = loss.mean()

        return loss 

    def supervised_loss(self, outputs, labels, task_labels): 

        if self.args.training_setting == 'task-il':
            labels = labels - (task_labels*self.cpt)

        supervised_loss = self.loss(outputs, labels)  
        return supervised_loss

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        ER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_tasklabels = self.buffer.get_data(
                self.args.minibatch_size, transform=None, device=self.device)
            buf_bsz = buf_inputs.shape[0]
            buf_inputs_1 = torch.cat([self.dataset.TRANSFORM(self.to_pil(b)).unsqueeze(0) for b in buf_inputs],dim=0)
            buf_inputs_2 = torch.cat([self.dataset.TRANSFORM(self.to_pil(b)).unsqueeze(0) for b in buf_inputs],dim=0)
            buf_inputs_1 = buf_inputs_1.to(self.device); buf_inputs_2 = buf_inputs_2.to(self.device)
            outputs_buf, features_buf = self.net(torch.cat([buf_inputs_1,buf_inputs_2], dim=0), task_label=buf_tasklabels if not self.args.training_setting == 'class-il' else None,returnt='supcon')
            f_buf_1, f_buf_2 = torch.split(features_buf, [buf_bsz, buf_bsz], dim=0)
            features_buf = torch.cat([f_buf_1.unsqueeze(1), f_buf_2.unsqueeze(1)], dim=1)
            
        bsz = labels.shape[0]

        if self.args.training_setting == 'class-il':
            task_labels = None
        else:  
            task_labels = torch.ones(labels.shape[0],  dtype=torch.int64, device=self.device) * self.current_task

        outputs, features = self.net(inputs, task_label=task_labels, returnt='supcon')

        if not self.buffer.is_empty():
            if self.args.training_setting == 'class-il': tl = None
            else: tl = torch.cat([task_labels, buf_tasklabels])
            supervised_loss = self.supervised_loss(torch.cat([outputs[:bsz], outputs_buf[:buf_bsz]]), torch.cat([labels, buf_labels]), tl)
        else:
            supervised_loss = self.supervised_loss(outputs[:bsz], labels, task_labels)

        # contrastive loss 
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if not self.buffer.is_empty():
            features = torch.cat([features, features_buf], dim=0)
            labels = torch.cat([labels, buf_labels], dim=0)
            bsz += self.args.minibatch_size
        contrastive_loss = self.contrastive_loss(features, labels, batch_size=bsz)

        loss= self.alpha * supervised_loss + contrastive_loss 
        loss.backward()
        self.opt.step()

        return loss.item()
