import os
import datetime

import numpy as np
import torch
import torch.nn as nn

class EBMAligner:
    """Manages the lifecycle of the proposed Energy Based Latent Alignment.
    """

    def __init__(self,
                 ebm_latent_dim=64,
                 ebm_n_layers=2,
                 ebm_n_hidden_units=64,
                 max_iter=1500,
                 ebm_lr=0.0001,
                 n_langevin_steps=30,
                 langevin_lr=0.1,
                 ema_decay=0.89,
                 log_iter=50,
                 output_dir='/home/joseph/workspace/lantern'):
        self.is_enabled = False

        # Configs of the EBM model
        self.ebm_latent_dim = ebm_latent_dim
        self.ebm_n_layers = ebm_n_layers
        self.ebm_n_hidden_units = ebm_n_hidden_units
        self.ebm_ema = None

        # EBM Learning configs
        self.max_iter = max_iter
        self.ebm_lr = ebm_lr
        self.n_langevin_steps = n_langevin_steps
        self.langevin_lr = langevin_lr
        self.ema_decay = ema_decay
        self.log_iter = log_iter
        self.output_dir = output_dir
        self.known_classes = 0

    def ema(self, model1, model2, decay=0.999):
        par1 = dict(model1.named_parameters())
        par2 = dict(model2.named_parameters())
        for k in par1.keys():
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def sampler(self, ebm_model, x, langevin_steps=30, lr=0.1, return_seq=False):
        """The langevin sampler to sample from the ebm_model
        :param ebm_model: The source EBM model
        :param x: The data which is updated to minimize energy from EBM
        :param langevin_steps: The number of langevin steps
        :param lr: The langevin learning rate
        :param return_seq: Whether to return the sequence of updates to x
        :return: Sample(s) from EBM
        """
        x = x.clone().detach()
        x.requires_grad_(True)
        sgd = torch.optim.SGD([x], lr=lr)

        sequence = torch.zeros_like(x).unsqueeze(0).repeat(langevin_steps, 1, 1)
        for k in range(langevin_steps):
            sequence[k] = x.data
            ebm_model.zero_grad()
            sgd.zero_grad()
            energy = ebm_model(x).sum()

            (-energy).backward()
            sgd.step()

        if return_seq:
            return sequence
        else:
            return x.clone().detach()

    def learn_ebm(self,
                  prev_model,
                  current_model,
                  current_task_dataloader,
                  exemplar_dataloader=None,
                  exemplar_plus_current_dataloader=None,
                  validation_data=None,
                  validation_data_all_class=None,
                  known_classes=None,
                  use_mixup=False,
                  b2_model=None,
                  ref_b2_model=None,
                  fusion_vars=None,
                  ref_fusion_vars=None,
                  args=None,
                  iteration=0,
                  start_iter=0,
                  task_id=0):
        """Learn the EBM.

        current_task_data + prev_model acts as in-distribution data, and
        current_task_data + current_model acts as out-of-distribution data.
        This is used for learning the energy manifold.

        :param prev_model: Model trained till previous task.
        :param current_model: Model trained on current task.
        :param current_task_data: Datapoints from the current incremental task.
        :param validation_data: OPTIONAL, if passed, used for evaluation.
        :return: None.
        """
        self.known_classes = known_classes
        print('Known classes: {}'.format(self.known_classes))
        ebm = EBM(latent_dim=self.ebm_latent_dim, n_layer=self.ebm_n_layers,
                  n_hidden=self.ebm_n_hidden_units).cuda()
        # if self.ebm_ema is None:
        self.ebm_ema = EBM(latent_dim=self.ebm_latent_dim, n_layer=self.ebm_n_layers,
                           n_hidden=self.ebm_n_hidden_units).cuda()
        # Initialize the exponential moving average of the EBM.
        self.ema(self.ebm_ema, ebm, decay=0.)

        ebm_optimizer = torch.optim.RMSprop(ebm.parameters(), lr=self.ebm_lr)

        iterations = 0
        prev_model.eval()
        current_model.eval()
        data_iter = iter(current_task_dataloader)
        # exemplar_iter = iter(exemplar_plus_current_dataloader)

        print('Starting to learn the EBM')
        while iterations < self.max_iter:
            ebm.zero_grad()
            ebm_optimizer.zero_grad()

            try:
                inputs = next(data_iter)
                # exemplars, _ = next(exemplar_iter)
            except (OSError, StopIteration):
                data_iter = iter(current_task_dataloader)
                inputs = next(data_iter)

            inputs = inputs[0].cuda()
            if use_mixup:
                alpha = np.random.beta(args.beta_param_1, args.beta_param_2)
            else:
                alpha = 0

            inputs_shuffled = inputs[torch.randperm(inputs.size()[0])]
            inputs = alpha * inputs_shuffled + (1 - alpha) * inputs

            prev_z = prev_model.net(inputs, 'features')
            current_z = current_model.net(inputs, 'features')

            self.requires_grad(ebm, False)
            sampled_z = self.sampler(ebm, current_z.clone().detach(), langevin_steps=self.n_langevin_steps,
                                     lr=self.langevin_lr)
            self.requires_grad(ebm, True)

            indistribution_energy = ebm(prev_z)
            oodistribution_energy = ebm(sampled_z)

            loss = -(indistribution_energy - oodistribution_energy).mean()

            loss.backward()
            ebm_optimizer.step()
            self.ema(self.ebm_ema, ebm, decay=self.ema_decay)

            if iterations == 0 or iterations % self.log_iter == 0:
                if validation_data is not None:
                    accuracy_prev, accuracy_curr, accuracy_curr_on_train_set = self.evaluate_after_selective_alignment_onpro(prev_model,
                                                                                           current_model,
                                                                                           validation_data,
                                                                                           validation_data_all_class,
                                                                                           b2_model,
                                                                                           fusion_vars,
                                                                                           args,
                                                                                           task_id)
                    # print("Iteration: {:5d}, Accuracy: {:5.2f}".format(iterations, accuracy_curr))
                    print("Iteration: {:5d}, Accuracy current: {:5.2f}, Loss: {:5.4f}".format(iterations, accuracy_curr, loss))
                else:
                    print("Iter: {:5d}".format(iterations))

            iterations += 1

        self.is_enabled = True

    def evaluate(self, previous_model, current_model, validation_data, validation_data_all_class):
        previous_model.eval()
        current_model.eval()
        accuracy_metric_prev_model = Metrics()
        accuracy_metric_current_model = Metrics()

        for inputs, labels in validation_data:
            inputs = inputs.cuda()
            labels = labels.cuda()
            current_z = current_model.net(inputs, 'features')
            aligned_z = self.align_latents(current_z)

            output = previous_model.get_output(aligned_z)
            accuracy_prev_model = self.compute_accuracy(output, labels)[0].item()
            accuracy_metric_prev_model.update(accuracy_prev_model)

        for inputs, labels in validation_data_all_class:
            inputs = inputs.cuda()
            labels = labels.cuda()
            current_z = current_model.net(inputs, 'features')
            aligned_z = self.align_latents(current_z)

            output = current_model.get_output(aligned_z)
            accuracy_curr_model = self.compute_accuracy(output, labels)[0].item()
            accuracy_metric_current_model.update(accuracy_curr_model)

        return accuracy_metric_prev_model.avg, accuracy_metric_current_model.avg

    def evaluate_after_selective_alignment(self, previous_model, current_model, validation_data,
                                           validation_data_all_class, b2_model, fusion_vars, args):
        previous_model.eval()
        current_model.eval()
        accuracy_metric_prev_model = Metrics()
        accuracy_metric_current_model = Metrics()

        for inputs, labels in validation_data:
            inputs = inputs.cuda()
            labels = labels.cuda()
            current_z = current_model.net(inputs, 'features')
            aligned_z = self.align_latents(current_z)

            output = previous_model.get_output(aligned_z)
            accuracy_prev_model = self.compute_accuracy(output, labels)[0].item()
            accuracy_metric_prev_model.update(accuracy_prev_model)

        for inputs, labels in validation_data_all_class:
            inputs = inputs.cuda()
            labels = labels.cuda()

            output, all_labels = self.align_prev_classes(inputs, labels, current_model, b2_model, fusion_vars, args)

            accuracy_curr_model = self.compute_accuracy(output, all_labels)[0].item()
            accuracy_metric_current_model.update(accuracy_curr_model)

        return accuracy_metric_prev_model.avg, accuracy_metric_current_model.avg

    def evaluate_after_selective_alignment_onpro(self, previous_model, current_model, validation_data,
                                           validation_data_all_class, b2_model, fusion_vars, args,task_id):
        previous_model.eval()
        current_model.eval()
        accuracy_metric_prev_model = Metrics()
        accuracy_metric_current_model = Metrics()
        accuracy_metric_current_model_on_train_set = Metrics()

        for i in range(task_id + 1):
            for inputs, labels in validation_data[i]['test']:
                inputs = inputs.cuda()
                labels = labels.cuda()
                current_z = current_model.net(inputs, 'features')

                if i != task_id:
                    aligned_z = self.align_latents(current_z)
                else:
                    aligned_z = current_z

                output = current_model.get_output(aligned_z)
                accuracy_curr_model = self.compute_accuracy(output, labels)[0].item()
                accuracy_metric_current_model.update(accuracy_curr_model)

        return accuracy_metric_prev_model.avg, accuracy_metric_current_model.avg, accuracy_metric_current_model_on_train_set.avg

    def compute_accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        result = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            result.append(correct_k.mul_(100.0 / batch_size))
        return result

    def align_latents(self, z):
        self.requires_grad(self.ebm_ema, False)
        aligned_z = self.sampler(self.ebm_ema, z.clone().detach(), langevin_steps=self.n_langevin_steps,
                                 lr=self.langevin_lr)
        self.requires_grad(self.ebm_ema, True)
        return aligned_z

    def align_prev_classes(self, inputs, labels, current_model, b2_model, fusion_vars, args, return_z=False):
        z = None
        all_labels = None

        # Split inputs and labels for current task
        mask = labels.ge(self.known_classes)  # Current task mask
        if True in mask:
            inputs_current_task = inputs[mask]
            all_labels = labels[mask]
            # if args.branch_mode == 'dual':
            #     z = process_inputs_fp(args, fusion_vars, current_model, b2_model, inputs_current_task,
            #                           feature_mode=True)
            # else:
            z = current_model.features(inputs_current_task)

        # Split inputs and labels for previous task and align them
        if True in ~mask:
            inputs_prev_tasks = inputs[~mask]
            labels_prev_tasks = labels[~mask]

            z_prev_tasks = current_model.net(inputs_prev_tasks, 'features')
            aligned_z_prev_tasks = self.align_latents(z_prev_tasks)
            if z is not None:
                z = torch.cat((z, aligned_z_prev_tasks))
                all_labels = torch.cat((all_labels, labels_prev_tasks))
            else:
                z = aligned_z_prev_tasks
                all_labels = labels_prev_tasks

        # Forward pass the latents through the classification head
        output = current_model.get_output(z)

        if return_z:
            return output, all_labels, z

        return output, all_labels


class EBM(nn.Module):
    """Defining the Energy Based Model.
    """

    def __init__(self, latent_dim=32, n_layer=1, n_hidden=64):
        super().__init__()

        mlp = nn.ModuleList()
        if n_layer == 0:
            mlp.append(nn.Linear(latent_dim, 1))
        else:
            mlp.append(nn.Linear(latent_dim, n_hidden))

            for _ in range(n_layer - 1):
                mlp.append(nn.LeakyReLU(0.2))
                mlp.append(nn.Linear(n_hidden, n_hidden))

            mlp.append(nn.LeakyReLU(0.2))
            mlp.append(nn.Linear(n_hidden, 1))

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)


class Metrics:
    def __init__(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
