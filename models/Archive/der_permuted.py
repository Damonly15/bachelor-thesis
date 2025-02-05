import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer, fill_buffer
from utils.metrics import get_pretrained

class DerPermuted(ContinualModel):
    NAME = 'der_permuted'
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
        super(DerPermuted, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)

        self.pretrained_model = get_pretrained(args)
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()

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

        self.opt.step()

        return tot_loss

    def end_task(self, dataset):
        with torch.no_grad():

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
                    inputs, labels, not_aug_inputs, _ = data
                else:
                    inputs, labels, not_aug_inputs = data
                    
                inputs = inputs.to(self.device)
                not_aug_inputs = not_aug_inputs.to(self.device)
                outputs = self.pretrained_model(inputs)
                outputs = outputs.detach()

                #permute outputs
                for pos in range(outputs.shape[0]):
                    fixed_pos = labels[pos].item()
                    sub_tensor = outputs[pos]
                    elements_to_permute = torch.cat((sub_tensor[:fixed_pos], sub_tensor[fixed_pos+1:]))
                    permuted_elements = elements_to_permute[torch.randperm(elements_to_permute.size(0))]
                    new_sub_tensor = torch.cat((permuted_elements[:fixed_pos], sub_tensor[fixed_pos:fixed_pos+1], permuted_elements[fixed_pos:]))
                    outputs[pos] = new_sub_tensor

                if self.args.temperature <= 100:
                    outputs = F.log_softmax(outputs / self.args.temperature, dim=-1)
                self.buffer.add_data(examples=not_aug_inputs[:(examples_per_task - counter)],
                                        logits=outputs.detach()[:(examples_per_task - counter)],
                                        task_labels=(torch.ones(self.args.batch_size, dtype=torch.long) *
                                                    current_task)[:(examples_per_task - counter)])
            
                counter += self.args.batch_size
                if examples_per_task - counter <= 0:
                    break