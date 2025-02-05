import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer, fill_buffer
from utils.batch_norm import bn_track_stats
from utils.model_utils import adjust_outputs

class DerBounds2(ContinualModel):
    NAME = 'der_bounds2'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il'] #this needs task boundaries but no test time oracle

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
        super(DerBounds2, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)
        self.buffer2 = Buffer(self.args.buffer_size)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        tot_loss = 0

        if self.args.training_setting == "class-il":
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
        else:
            outputs = self.net(inputs)
            outputs = adjust_outputs(outputs, (torch.ones(outputs.shape[0], dtype=torch.int64, device=self.device) * self.current_task), self._cpt)
            labels = labels - self.n_past_classes
            loss = self.loss(outputs, labels)
            loss.backward()
            tot_loss += loss.item()

            if not self.buffer.is_empty():
                buf_inputs, _, buf_logits, buf_tasklabels = self.buffer.get_data(
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

    def end_task(self, dataset): #Changed this for the paper, it is from xder. It makes sure, that every class has the same amount of samples in the buffer.
        tng = self.net.training
        self.net.train()

        examples_per_class = self.args.buffer_size // ((self.current_task + 1) * self.cpt)
        remainder = self.args.buffer_size % ((self.current_task + 1) * self.cpt)
        remainder2 = remainder

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

            #second buffer
            buf_x, buf_lab, buf_log, buf_tl = self.buffer2.get_all_data()
            self.buffer2.empty()

            for tl in buf_lab.unique():
                idx = tl == buf_lab
                ex, lab, log, tasklab = buf_x[idx], buf_lab[idx], buf_log[idx], buf_tl[idx]
                if(remainder2 > 0):
                    first = min(ex.shape[0], examples_per_class + 1)
                    remainder2 -= 1
                else:
                    first = min(ex.shape[0], examples_per_class)
                self.buffer2.add_data(
                    examples=ex[:first],
                    labels=lab[:first],
                    logits=log[:first],
                    task_labels=tasklab[:first]
                )

        # fdr add new task
        ce = torch.tensor([examples_per_class] * self.cpt).int()
        for i in range(remainder):
            ce[i] += 1 
        
        ce2 = torch.tensor([examples_per_class] * self.cpt).int()
        for i in range(remainder2):
            ce2[i] += 1 

        with torch.no_grad():
            with bn_track_stats(self.net, False):
                for data in dataset.train_loader:
                    inputs, labels, not_aug_inputs = data
                    inputs = inputs.to(self.device)
                    outputs = self.net(inputs)

                    if self.args.training_setting ==  "class-il":
                        if self.args.temperature <= 100:
                            outputs = F.log_softmax(outputs / self.args.temperature, dim=-1)
                    else:
                        outputs = adjust_outputs(outputs, (torch.ones(outputs.shape[0], dtype=torch.int64, device=self.device) * self.current_task), self._cpt)
                        if self.args.temperature <= 100:
                            outputs = F.log_softmax(outputs / self.args.temperature, dim=-1)

                    if not all(ce == 0):
                        flags = torch.zeros(len(inputs)).bool()
                        for j in range(len(flags)):
                            if ce[labels[j] % self.cpt] > 0:
                                flags[j] = True
                                ce[labels[j] % self.cpt] -= 1

                        self.buffer.add_data(examples=not_aug_inputs[flags],
                                            labels=labels[flags],
                                            logits=((outputs.detach()).cpu())[flags],
                                            task_labels=(torch.ones(len(flags), dtype=torch.int64) * self.current_task)[flags])
                    elif not all(ce2 == 0):
                        flags = torch.zeros(len(inputs)).bool()
                        for j in range(len(flags)):
                            if ce2[labels[j] % self.cpt] > 0:
                                flags[j] = True
                                ce2[labels[j] % self.cpt] -= 1

                        self.buffer2.add_data(examples=not_aug_inputs[flags],
                                            labels=labels[flags],
                                            logits=((outputs.detach()).cpu())[flags],
                                            task_labels=(torch.ones(len(flags), dtype=torch.int64) * self.current_task)[flags])
                    else:
                        break
         
        self.net.train(tng)
