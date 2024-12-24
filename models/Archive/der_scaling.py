import torch
from torch.nn import functional as F
import torch.optim as optim
import copy

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer, fill_buffer
from utils.batch_norm import bn_track_stats
from utils.feature_forgetting import evaluate_previous
from utils.conf import base_path
from utils import create_if_not_exists

class DerScaling(ContinualModel):
    NAME = 'der_scaling'
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
        super(DerScaling, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)
        self.net2 = None
        self.opt2 = None

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

                if self.net2 is not None:
                    self.opt2.zero_grad()
                    buf_outputs2 = self.net2(buf_inputs)
                    buf_outputs2 = F.log_softmax(buf_outputs2 / self.args.temperature, dim=-1)
                    loss_kl2 = self.args.alpha * self.args.temperature**2 * F.kl_div(buf_outputs2, buf_logits, reduction='batchmean', log_target=True)
                    loss_kl2.backward()
                    tot_loss += loss_kl2.item()
                    self.opt2.step()
            else:
                loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
                loss_mse.backward()
                tot_loss += loss_mse.item()

                if self.net2 is not None:
                    self.opt2.zero_grad()
                    buf_outputs2 = self.net2(buf_inputs)
                    loss_mse2 = self.args.alpha * F.mse_loss(buf_outputs2, buf_logits)
                    loss_mse2.backward()
                    tot_loss += loss_mse2.item()
                    self.opt2.step()
 
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
        
        if self.current_task + 2 == self.N_TASKS:
            #second last task, copy network
            self.net2 = copy.deepcopy(self.net)
            self.opt2 = optim.SGD(self.net2.parameters(), lr=self.args.lr,
                                    weight_decay=self.args.optim_wd,
                                    momentum=self.args.optim_mom,
                                    nesterov=self.args.optim_nesterov == 1)

        elif self.current_task + 1 == self.N_TASKS:
            #last task, evaluate second model
            accuracy, _ = evaluate_previous(self.net2, dataset, self.device)
            mean_accuracy = sum(accuracy) / len(accuracy)

            wrargs = (vars(self.args)).copy()
            target_folder = base_path() + "results/"

            #here we log to class-il accuracies
            wrargs['accmean_task' + str(dataset.N_TASKS)] = mean_accuracy

            for i, fa in enumerate(accuracy):
                wrargs['accuracy_' + str(i + 1) + '_task' + str(dataset.N_TASKS)] = fa

            create_if_not_exists(target_folder + self.args.training_setting)
            create_if_not_exists(target_folder + self.args.training_setting +
                                "/" + self.args.dataset)
            create_if_not_exists(target_folder + self.args.training_setting +
                                "/" + self.args.dataset + "/der_scaling_buffer")

            path = target_folder + self.args.training_setting + "/" + self.args.dataset\
                + "/der_scaling_buffer/logs.txt"
            print("Logging Class-IL results and arguments in " + path)
            with open(path, 'a') as f:
                f.write(str(wrargs) + '\n')
