import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer, fill_buffer
from utils.metrics import adjust_outputs
from utils.batch_norm import bn_track_stats

class DerFeature(ContinualModel):
    NAME = 'der_feature'
    #this needs task boundaries but no test time oracle
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via'
                                ' Dark Experience Replay with KL loss and task boundaries.')
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        parser.add_argument('--beta', type=float, required=True,
                            help='Penalty weight.')
        
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(DerFeature, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.opt.zero_grad()
        tot_loss = 0

        if self.args.training_setting == "class-il":
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            tot_loss += loss.item()

            if not self.buffer.is_empty():
                buf_inputs, _, buf_features, _ = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, device=self.device)
                buf_outputs = self.net.forward(buf_inputs, returnt="features")
                loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_features)
                loss_mse.backward()
                tot_loss += loss_mse.item()

                buf_inputs, buf_labels, _, _ = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, device=self.device)
                buf_outputs = self.net(buf_inputs)
                loss_ce = self.args.beta * self.loss(buf_outputs, buf_labels)
                loss_ce.backward()
                tot_loss += loss_ce.item()
        else:
            outputs = self.net(inputs)
            outputs = adjust_outputs(outputs, (torch.ones(outputs.shape[0], dtype=torch.long, device=self.device) * self.current_task), self._cpt)
            labels = labels - self.n_past_classes
            loss = self.loss(outputs, labels)
            loss.backward()
            tot_loss += loss.item()

            if not self.buffer.is_empty():
                buf_inputs, _, buf_features, _ = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, device=self.device)
                buf_outputs = self.net.forward(buf_inputs, returnt="features")
                loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_features)
                loss_mse.backward()
                tot_loss += loss_mse.item()
                
                buf_inputs, buf_labels, _, buf_tasklabels = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, device=self.device)
                buf_outputs = self.net(buf_inputs)
                buf_outputs = adjust_outputs(buf_outputs, buf_tasklabels, self._cpt)
                loss_ce = self.args.beta * self.loss(buf_outputs, buf_labels)
                loss_ce.backward()
                tot_loss += loss_ce.item()

        self.opt.step()

        return tot_loss

    def end_task(self, dataset):
        with torch.no_grad():
            #deactivate running mean and variance of batchnorm layers
            with bn_track_stats(self.net, False):
                #make space in the buffer (each task has the same amount of samples in the buffer)
                current_task = self.current_task
                examples_per_task = self.args.buffer_size // (current_task+1)
                remainder = self.args.buffer_size % (current_task+1)

                if(not self.buffer.is_empty()):
                    buf_x, buf_lab, buf_log, buf_tl = self.buffer.get_all_data()
                    self.buffer.empty()

                    for ttl in buf_tl.unique():
                        idx = (buf_tl == ttl)
                        ex, lab, log, tasklab = buf_x[idx], buf_lab[idx], buf_log[idx], buf_tl[idx]
                        if(remainder > 0):
                            first = min(ex.shape[0], examples_per_task + 1)
                            remainder -= 1
                        else:
                            first = min(ex.shape[0], examples_per_task)

                        self.buffer.add_data(
                            examples=ex[:first],
                            labels=lab[:first],
                            logits=log[:first],
                            task_labels=tasklab[:first])

                #do some foreward passes to fill up buffer with samples from the current task
                counter = 0
                for data in dataset.train_loader:
                    if hasattr(dataset.train_loader.dataset, 'logits'):
                        inputs, labels, not_aug_inputs, _ = data
                    else:
                        inputs, labels, not_aug_inputs = data   

                    inputs = inputs.to(self.device)
                    not_aug_inputs = not_aug_inputs.to(self.device)
                    features = self.net.forward(inputs, returnt="features")

                    if self.args.training_setting == "task-il":
                        labels = labels - self.n_past_classes
                                
                    self.buffer.add_data(examples=not_aug_inputs[:(examples_per_task - counter)],
                                            labels=labels[:(examples_per_task - counter)],
                                            logits=features.detach()[:(examples_per_task - counter)],
                                            task_labels=(torch.ones(self.args.batch_size, dtype=torch.long) *
                                                        current_task)[:(examples_per_task - counter)])

                    counter += self.args.batch_size
                    if examples_per_task - counter <= 0:
                        break