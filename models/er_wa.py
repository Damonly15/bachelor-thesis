import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer, fill_buffer

class ErWA(ContinualModel):
    NAME = 'er_wa'
    COMPATIBILITY = ['class-il'] #this needs task boundaries but no test time oracle

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via'
                                'Experience Replay with task boundaries. Before evaluation, is applies weight aligning.')
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(ErWA, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)

        self.scaling_factor = 1

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()

        if self.args.training_setting == 'class-il':
            task_labels = None
        else: 
            task_labels = torch.ones(labels.shape[0],  dtype=torch.int64, device=self.device) * self.current_task
            labels = labels - (task_labels*self.cpt)

        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_tasklabels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            
            if self.args.training_setting == 'task-il':
                buf_labels = buf_labels - (buf_tasklabels*self.cpt)
                task_labels = torch.cat((task_labels, buf_tasklabels), dim=0)
            inputs = torch.cat((inputs, buf_inputs), dim=0)
            labels = torch.cat((labels, buf_labels), dim=0)

        outputs = self.net.forward(inputs, task_label=task_labels)
        loss = self.loss(outputs, labels)
        loss.backward()
                      
        self.opt.step()

        #weight clipping to be positive
        with torch.no_grad():
            self.net.classifier.weight.data.clamp_(min=0)
            if self.net.classifier.bias is not None:  # Check if the layer has a bias term
                assert(False)
                self.net.classifier.bias.data.clamp_(min=0)  # Clip bias to be non-negative

            '''
            _wandb_old_position = self.current_task * self._cpt
            _wandb_new_position = _wandb_old_position + self._cpt

            if _wandb_old_position > 0:
                norms_old = self.net.classifier.weight[0:_wandb_old_position].norm(p=2, dim=1)
                _wandb_mean_norm_old = norms_old.mean()

            norms_new = self.net.classifier.weight[_wandb_old_position:_wandb_new_position].norm(p=2, dim=1)
            _wandb_mean_norm_new = norms_new.mean()
            '''

        return loss.item()

    def end_task(self, dataset): #Changed this for the paper, it is from xder. It makes sure, that every class has the same amount of samples in the buffer.
        examples_per_class = self.args.buffer_size // ((self.current_task + 1) * self.cpt)
        remainder = self.args.buffer_size % ((self.current_task + 1) * self.cpt)
        ones_indices = torch.randperm(self.n_seen_classes)[:remainder]
        remainder = torch.zeros(self.n_seen_classes)
        if not self.args.buffer_size == dataset.N_CLASSES: #in this case just use one sample per class
            remainder[ones_indices] = 1

        # fdr reduce coreset
        if not self.buffer.is_empty():
            buf_x, buf_lab, buf_tl = self.buffer.get_all_data()
            self.buffer.empty()

            for tl in buf_lab.unique():
                idx = tl == buf_lab
                ex, lab, tasklab = buf_x[idx], buf_lab[idx], buf_tl[idx]
                first = min(ex.shape[0], examples_per_class + int(remainder[tl].item()))
                self.buffer.add_data(
                    examples=ex[:first],
                    labels=lab[:first],
                    task_labels=tasklab[:first]
                )

        # fdr add new task
        ce = torch.tensor([examples_per_class] * self.cpt)
        ce = (ce + remainder[self.n_past_classes:]).int() 

        for data in dataset.train_loader:
            inputs, labels, not_aug_inputs = data
            if all(ce == 0):
                break

            flags = torch.zeros(len(inputs)).bool()
            for j in range(len(flags)):
                if ce[labels[j] % self.cpt] > 0:
                    flags[j] = True
                    ce[labels[j] % self.cpt] -= 1

            self.buffer.add_data(examples=not_aug_inputs[flags],
                                    labels=labels[flags],
                                    task_labels=(torch.ones(len(flags), dtype=torch.int64) * self.current_task)[flags])

        #weight aligning
        if self.current_task > 0:
            with torch.no_grad():
                #old task
                norms_old = self.net.classifier.weight[:self.n_past_classes].norm(p=2, dim=1)
                mean_norm_old = norms_old.mean()
                print(mean_norm_old)

                #current task
                norms_new = self.net.classifier.weight[self.n_past_classes:self.n_seen_classes].norm(p=2, dim=1)
                mean_norm_new = norms_new.mean()
                print(mean_norm_new)

                self.scaling = (mean_norm_old / mean_norm_new)
                self.net.classifier.weight[self.n_past_classes:self.n_seen_classes] =  self.scaling * self.net.classifier.weight[self.n_past_classes:self.n_seen_classes]
