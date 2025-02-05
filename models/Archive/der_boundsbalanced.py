from datasets.utils.continual_dataset import ContinualDataset
import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer, fill_buffer
from utils.batch_norm import bn_track_stats
from utils.metrics import adjust_outputs

class DerBoundsBalanced(ContinualModel):
    NAME = 'der_boundsbalanced'
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
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(DerBoundsBalanced, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)
        self.overall_batch_size = self.args.batch_size + self.args.minibatch_size
        

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        tot_loss = 0
        true_batch_size = inputs.shape[0]

        buf_inputs, buf_labels = [], []
        for i in range(self.current_task):
            current_inputs, current_logits, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device) 
            buf_inputs.append(current_inputs)
            buf_labels.append(current_logits)
        buf_inputs = torch.cat([inputs] + buf_inputs, dim=0)
        if(self.current_task > 0):
            buf_labels = torch.cat(buf_labels, dim=0)

        outputs = self.net(buf_inputs)
        loss = self.loss(outputs[0:true_batch_size], labels)
        loss.backward(retain_graph=True)
        tot_loss += loss.item()

        for i in range(self.current_task):
            start = true_batch_size + i*self.args.minibatch_size
            end = true_batch_size + (i+1)*self.args.minibatch_size
            buf_outputs = outputs[start:end]
            buf_logits = buf_labels[start-true_batch_size:end-true_batch_size]
            if self.args.temperature <= 100:
                buf_outputs = F.log_softmax(buf_outputs / self.args.temperature, dim=-1)
                loss_kl = self.args.alpha * self.args.temperature**2 * F.kl_div(buf_outputs, buf_logits, reduction='batchmean', log_target=True)
                if(i+1 != self.current_task):
                    loss_kl.backward(retain_graph=True)
                else:
                    loss_kl.backward()
                tot_loss += loss_kl.item()
            else:
                loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
                if(i+1 != self.current_task):
                    loss_mse.backward(retain_graph=True)
                else:
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
                    
                
                    if self.args.temperature <= 100:
                        outputs = F.log_softmax(outputs / self.args.temperature, dim=-1)
                    self.buffer.add_data(examples=not_aug_inputs[:(examples_per_task - counter)],
                                            logits=outputs.detach()[:(examples_per_task - counter)],
                                            task_labels=(torch.ones(self.args.batch_size, dtype=torch.long) *
                                                        current_task)[:(examples_per_task - counter)])
                    counter += self.args.batch_size
                    if examples_per_task - counter <= 0:
                        break

        #change batch size bevor starting training
        self.args.batch_size = self.overall_batch_size // (current_task+2)
        self.args.minibatch_size = self.overall_batch_size // (current_task+2)

        #also adjust lr to account for more batches
        new_learning_rate = (self.args.lr / (current_task+2)) * 2
        for param_group in self.opt.param_groups:
            param_group['lr'] = new_learning_rate