import torch
from torch.nn import functional as F
from torch.optim import Adam

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer, fill_buffer

class ErBiC(ContinualModel):
    NAME = 'er_bic'
    COMPATIBILITY = ['class-il'] #this needs task boundaries but no test time oracle

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via'
                                ' Experience Replay with task boundaries. Before evaluation, is applies applies bias correction.')
        add_rehearsal_args(parser)
        parser.add_argument('--bic_iters', type=int, default=1000,
            help='bias injector.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(ErBiC, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)

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

        #bias correction
        if self.current_task > 0:
            status = self.net.training
            self.net.eval()

            corr_factors = torch.tensor([0., 1.], device=self.device, requires_grad=True)
            self.biasopt = Adam([corr_factors], lr=0.001)

            for l in range(self.args.bic_iters):
                buf_inputs, buf_labels, _ = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, device=self.device)

                self.biasopt.zero_grad()
                with torch.no_grad():
                    out = self.net.forward(buf_inputs)

                start_last_task = self.n_past_classes
                end_last_task = self.n_seen_classes
                tout = out
                tout[:, start_last_task:end_last_task] *= corr_factors[1].repeat_interleave(end_last_task - start_last_task)
                tout[:, start_last_task:end_last_task] += corr_factors[0].repeat_interleave(end_last_task - start_last_task)

                loss_bic = self.loss(tout[:, :end_last_task], buf_labels)
                loss_bic.backward()
                self.biasopt.step()

            self.corr_factors = corr_factors
            print(self.corr_factors)

            self.net.train(status)
    
    def forward(self, x):
        ret = self.net.forward(x)
        if ret.shape[0] > 0:
            if hasattr(self, 'corr_factors'):
                start_last_task = self.n_past_classes
                end_last_task = self.n_seen_classes
                ret[:, start_last_task:end_last_task] *= self.corr_factors[1].repeat_interleave(end_last_task - start_last_task)
                ret[:, start_last_task:end_last_task] += self.corr_factors[0].repeat_interleave(end_last_task - start_last_task)
        return ret