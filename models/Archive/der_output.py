import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer, fill_buffer
from utils.batch_norm import bn_track_stats
from utils.model_utils import adjust_outputs

class DerOutput(ContinualModel):
    NAME = 'der_output'
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
                            help='Prior penalty weight.')
        parser.add_argument('--temperature', type=float, required=True,
                            help='Temperature of softmax.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(DerOutput, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)
        self.prior_weights = None
        self.prior_biases = None

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        tot_loss = 0

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

            prior_loss = self.args.beta * self.prior_loss()
            prior_loss.backward()
            tot_loss += prior_loss.item()

        if epoch < 5 and self.current_task != 0:
            with torch.no_grad():
                self.net.classifier.weight.grad[:self.current_task * self._cpt] = 0
                self.net.classifier.bias.grad[:self.current_task * self._cpt] = 0

        self.opt.step()

        return tot_loss

    def end_task(self, dataset):
        current_task = self.current_task
        if (current_task == 0):
            self.prior_weights = self.net.classifier.weight[:self._cpt, :].detach().clone()
            self.prior_biases = self.net.classifier.bias[:self._cpt].detach().clone()
        else:
            start = current_task*self._cpt
            end = start + self._cpt
            current_weights = self.net.classifier.weight[start:end, :].detach().clone()
            current_biases = self.net.classifier.bias[start:end].detach().clone()
            self.prior_weights = torch.cat([self.prior_weights, current_weights], dim=0)
            self.prior_biases = torch.cat([self.prior_biases, current_biases], dim=0)

        with torch.no_grad():
            #deactivate running mean and variance of batchnorm layers
            with bn_track_stats(self.net, False):
                #make space in the buffer (each task has the same amount of samples in the buffer)
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
                    
                    if self.args.temperature <= 100:
                        outputs = F.log_softmax(outputs / self.args.temperature, dim=-1)
                    self.buffer.add_data(examples=not_aug_inputs[:(examples_per_task - counter)],
                                            logits=outputs.detach()[:(examples_per_task - counter)],
                                            task_labels=(torch.ones(self.args.batch_size, dtype=torch.long) *
                                                        current_task)[:(examples_per_task - counter)])
                    counter += self.args.batch_size
                    if examples_per_task - counter <= 0:
                        break

    def prior_loss(self):
        total_weights = self.prior_weights.numel()
        total_biases = self.prior_biases.numel()

        end = self.current_task * self._cpt
        current_weights = self.net.classifier.weight[:end, :]
        current_biases = self.net.classifier.bias[:end]
        loss_weights = F.mse_loss(self.prior_weights, current_weights, reduction="sum")
        loss_bias = F.mse_loss(self.prior_biases, current_biases, reduction="sum")

        return loss_weights / total_weights + loss_bias / total_biases