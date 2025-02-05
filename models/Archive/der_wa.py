import torch
import torch.nn as nn
from torch.nn import functional as F
import copy

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer, fill_buffer
from utils.batch_norm import bn_track_stats

class DerWA(ContinualModel):
    NAME = 'der_wa'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il'] #this needs task boundaries but no test time oracle

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via'
                                ' Dark Experience Replay with KL loss and task boundaries. Before evaluation, is applies weight aligning.')
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        parser.add_argument('--temperature', type=float, required=True,
                            help='Temperature of softmax.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(DerWA, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)

        self.scaling_factor = 1

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        tot_loss = 0

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        tot_loss += loss.item()

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            buf_outputs = self.net(buf_inputs)

            if self.args.temperature <= 100:
                buf_outputs = F.log_softmax(buf_outputs / self.args.temperature, dim=-1)
                loss_kl = self.args.alpha * self.args.temperature**2 * F.kl_div(buf_outputs, buf_logits, reduction='batchmean', log_target=True)
                loss_kl.backward()
                tot_loss += loss_kl.item()
            else:
                loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
                loss_mse.backward()
                tot_loss += loss_mse.item()

        self.opt.step()

        #weight clipping to be positive
        with torch.no_grad():
            self.net.classifier.weight.data.clamp_(min=0)
            if self.net.classifier.bias is not None:  # Check if the layer has a bias term
                self.net.classifier.bias.data.clamp_(min=0)  # Clip bias to be non-negative

            _wandb_old_position = self.current_task * self._cpt
            _wandb_new_position = _wandb_old_position + self._cpt

            if _wandb_old_position > 0:
                norms_old = self.net.classifier.weight[0:_wandb_old_position].norm(p=2, dim=1)
                _wandb_mean_norm_old = norms_old.mean()

            norms_new = self.net.classifier.weight[_wandb_old_position:_wandb_new_position].norm(p=2, dim=1)
            _wandb_mean_norm_new = norms_new.mean()

        return tot_loss

    def begin_task(self, dataset):
        if self.current_task > 1:
            old_position = (self.current_task-1) * self._cpt
            new_position = old_position + self._cpt
            print(f'aligning_{old_position}_{new_position}')

            with torch.no_grad(): 
                self.net.classifier.weight[old_position:new_position] =  (1 / self.scaling) * self.net.classifier.weight[old_position:new_position]

    def end_task(self, dataset): #Changed this for the paper, it is from xder. It makes sure, that every class has the same amount of samples in the buffer.
        tng = self.net.training
        self.net.train()

        if self.args.buffer_size == dataset.N_CLASSES: #one sample per class
            examples_per_class = 1
            remainder = 0
        else:
            examples_per_class = self.args.buffer_size // ((self.current_task + 1) * self.cpt)
            remainder = self.args.buffer_size % ((self.current_task + 1) * self.cpt)

        # fdr reduce coreset
        if not self.buffer.is_empty():
            buf_x, buf_lab, buf_log, buf_tl = self.buffer.get_all_data()
            self.buffer.empty()

            for tl in buf_lab.unique():
                idx = tl == buf_lab
                ex, lab, log, tasklab = buf_x[idx], buf_lab[idx], buf_log[idx], buf_tl[idx]
                if(remainder > 0):
                    first = min(ex.shape[0], examples_per_class + 1)
                    remainder -= 1
                else:
                    first = min(ex.shape[0], examples_per_class)
                self.buffer.add_data(
                    examples=ex[:first],
                    labels=lab[:first],
                    logits=log[:first],
                    task_labels=tasklab[:first]
                )

        # fdr add new task
        ce = torch.tensor([examples_per_class] * self.cpt).int()
        for i in range(remainder):
            ce[i] += 1 

        with torch.no_grad():
            with bn_track_stats(self.net, False):
                for data in dataset.train_loader:
                    inputs, labels, not_aug_inputs = data
                    inputs = inputs.to(self.device)
                    outputs = self.net(inputs)
                    if self.args.temperature <= 100:
                        outputs = F.log_softmax(outputs / self.args.temperature, dim=-1)

                    if all(ce == 0):
                        break

                    flags = torch.zeros(len(inputs)).bool()
                    for j in range(len(flags)):
                        if ce[labels[j] % self.cpt] > 0:
                            flags[j] = True
                            ce[labels[j] % self.cpt] -= 1

                    self.buffer.add_data(examples=not_aug_inputs[flags],
                                         labels=labels[flags],
                                         logits=((outputs.detach()).cpu())[flags],
                                         task_labels=(torch.ones(len(flags), dtype=torch.int64) * self.current_task)[flags])

        self.net.train(tng)

        #bias correction
        if self.current_task > 0:
            with torch.no_grad():
                old_position = self.current_task * self._cpt
                new_position = old_position + self._cpt
                print(f'aligning_{old_position}_{new_position}')

                norms_old = self.net.classifier.weight[0:old_position].norm(p=2, dim=1)
                mean_norm_old = norms_old.mean()

                norms_new = self.net.classifier.weight[old_position:new_position].norm(p=2, dim=1)
                mean_norm_new = norms_new.mean()

                self.scaling = (mean_norm_old / mean_norm_new)
                self.net.classifier.weight[old_position:new_position] =  self.scaling * self.net.classifier.weight[old_position:new_position]
