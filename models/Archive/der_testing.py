import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer, fill_buffer
from utils.batch_norm import bn_track_stats
from utils.model_utils import adjust_outputs

class DerTesting(ContinualModel):
    NAME = 'der_testing'
    #this needs task boundaries but no test time oracle
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via'
                                ' Dark Experience Replay with KL loss and task boundaries.')
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        parser.add_argument('--temperature', type=float, required=True,
                            help='Temperature of softmax.')
        parser.add_argument('--version', type=str, required=True,
                            help='Temperature of softmax.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(DerTesting, self).__init__(backbone, loss, args, transform)
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
                buf_inputs, buf_logits, _ = self.buffer.get_data(
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
        else:
            outputs = self.net(inputs)
            outputs = adjust_outputs(outputs, (torch.ones(outputs.shape[0], dtype=torch.long, device=self.device) * self.current_task), self._cpt)
            labels = labels - self.n_past_classes
            loss = self.loss(outputs, labels)
            loss.backward()
            tot_loss += loss.item()

            if not self.buffer.is_empty():
                buf_inputs, buf_logits, buf_tasklabels = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, device=self.device)
                buf_outputs = self.net(buf_inputs)
                buf_outputs = adjust_outputs(buf_outputs, buf_tasklabels, self._cpt)

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
                    buf_x, buf_log, buf_tl = self.buffer.get_all_data()
                    self.buffer.empty()

                    for ttl in buf_tl.unique():
                        idx = (buf_tl == ttl)
                        ex, log, tasklab = buf_x[idx], buf_log[idx], buf_tl[idx]
                        if(remainder > 0):
                            first = min(ex.shape[0], examples_per_task + 1)
                            remainder -= 1
                        else:
                            first = min(ex.shape[0], examples_per_task)

                        self.buffer.add_data(
                            examples=ex[:first],
                            logits=log[:first],
                            task_labels=tasklab[:first])

                #do some foreward passes to fill up buffer with samples from the current task
                counter = 0
                for data in dataset.train_loader:
                    if hasattr(dataset.train_loader.dataset, 'logits'):
                        inputs, _, not_aug_inputs, _ = data
                    else:
                        inputs, _, not_aug_inputs = data
                        
                    inputs = inputs.to(self.device)
                    not_aug_inputs = not_aug_inputs.to(self.device)
                    outputs = self.net(inputs)
                    
                    if self.args.training_setting ==  "class-il":
                        if self.args.temperature <= 100:
                            outputs = F.log_softmax(outputs / self.args.temperature, dim=-1)
                        self.buffer.add_data(examples=not_aug_inputs[:(examples_per_task - counter)],
                                                logits=outputs.detach()[:(examples_per_task - counter)],
                                                task_labels=(torch.ones(self.args.batch_size, dtype=torch.long) *
                                                            current_task)[:(examples_per_task - counter)])
                    else:
                        outputs = adjust_outputs(outputs, (torch.ones(outputs.shape[0], dtype=torch.long, device=self.device) * self.current_task), self._cpt)
                        if self.args.temperature <= 100:
                            outputs = F.log_softmax(outputs / self.args.temperature, dim=-1)
                        self.buffer.add_data(examples=not_aug_inputs[:(examples_per_task - counter)],
                                                logits=outputs.detach()[:(examples_per_task - counter)],
                                                task_labels=(torch.ones(self.args.batch_size, dtype=torch.long) *
                                                            current_task)[:(examples_per_task - counter)])

                    counter += self.args.batch_size
                    if examples_per_task - counter <= 0:
                        break
        
        if self.current_task + 1 != self.N_TASKS:
            return
                        
        if self.args.version == 'classifier':
            for param in self.net.parameters():
                param.requires_grad = False
            for param in self.net.classifier.parameters():
                param.requires_grad = True

        for i in range(15000):
            self.opt.zero_grad()
            buf_inputs, buf_logits, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            buf_outputs = self.net(buf_inputs)

            if self.args.temperature <= 100:
                buf_outputs = F.log_softmax(buf_outputs / self.args.temperature, dim=-1)
                loss_kl = self.args.alpha * self.args.temperature**2 * F.kl_div(buf_outputs, buf_logits, reduction='batchmean', log_target=True)
                loss_kl.backward()
            else:
                loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
                loss_mse.backward()
            self.opt.step()
        
        if self.args.version == 'classifier':
            for param in self.net.parameters():
                param.requires_grad = True
