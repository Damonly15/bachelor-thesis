import math
import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer


class DerBoundsFix(ContinualModel):
    NAME = 'der_boundsfix'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via'
                                ' Dark Experience Replay with task boundaries.')
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(DerBoundsFix, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        #Depending on the dataset, the last batch is of smaller size. This might be bad for this algorithm
        if(inputs.shape[0] != self.args.batch_size):
            return 0

        self.opt.zero_grad()
        tot_loss = 0

        if self.buffer.is_empty():
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            tot_loss += loss.item()

        else:
            buf_inputs, buf_logits, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)

            conc_inputs = torch.cat([inputs, buf_inputs], dim=0).to(self.device)
            conc_outputs = self.net(conc_inputs)
            
            loss = self.loss(conc_outputs[:inputs.shape[0]], labels)
            loss.backward(retain_graph=True)     #retain computational graph
            tot_loss += loss.item()

            #training acc
            _, pred = torch.max(conc_outputs[:inputs.shape[0], :self.n_seen_classes].data, 1)
            correct = torch.sum(pred == labels).item()
            total = labels.shape[0]
            _wandb_train_acc = correct / total * 100


            loss_mse = self.args.alpha * F.mse_loss(conc_outputs[-self.args.minibatch_size:], buf_logits)
            loss_mse.backward()
            tot_loss += loss_mse.item()

        self.opt.step()
        return tot_loss

    def end_task(self, dataset):
        with torch.no_grad():
            #deactivate running mean and variance of batchnorm layers
            for module in self.net.modules():
                if (isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d)):
                    module.track_running_stats = False

            #make space in the buffer (each task has the same amount of samples in the buffer)
            current_task = self.current_task + 1
            examples_per_task = self.args.buffer_size // current_task

            if(not self.buffer.is_empty()):
                buf_x, buf_log, buf_tl = self.buffer.get_all_data()
                self.buffer.empty()

                for ttl in buf_tl.unique():
                    idx = (buf_tl == ttl)
                    ex, log, tasklab = buf_x[idx], buf_log[idx], buf_tl[idx]
                    first = min(ex.shape[0], examples_per_task)
                    self.buffer.add_data(
                        examples=ex[:first],
                        logits=log[:first],
                        task_labels=tasklab[:first]
                    )

            #do some foreward passes to fill up buffer with samples from the current task
            interleave_inputs = None
            if(not self.buffer.is_empty()):
                interleave_inputs, _, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            
            counter = 0
            for data in dataset.train_loader:
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, _, not_aug_inputs, _ = data
                else:
                    inputs, _, not_aug_inputs = data
                
                inputs = inputs.to(self.device)
                if(not (interleave_inputs is None)):
                    interleave_inputs = interleave_inputs.to(self.device)
                    inputs = torch.cat([inputs, interleave_inputs],dim=0)

                inputs = inputs.to(self.device)
                not_aug_inputs = not_aug_inputs.to(self.device)
                outputs = self.net(inputs)
                outputs = outputs[:not_aug_inputs.shape[0]]
                self.buffer.add_data(examples=not_aug_inputs[:(examples_per_task - counter)],
                                        logits=outputs.data[:(examples_per_task - counter)],
                                        task_labels=(torch.ones(self.args.batch_size) *
                                                    current_task)[:(examples_per_task - counter)])
            
                counter += self.args.batch_size
                if examples_per_task - counter <= 0:
                    break

            #activate running mean and variance of batchnorm layers
            for module in self.net.modules():
                if (isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d)):
                    module.track_running_stats = True  