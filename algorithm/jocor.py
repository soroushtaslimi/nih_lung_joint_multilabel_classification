# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# from torchvision import models
from torch.nn import DataParallel
from model.resnet import *
from model.multihead_resnet import Multihead_Resnet
from sklearn.metrics import roc_auc_score

from model.cnn import MLPNet,CNN
import numpy as np
from common.utils import accuracy

from algorithm.loss import loss_jocor


class JoCoR:
    def __init__(self, args, train_dataset, device, input_channel, num_classes, num_test_samples, start_checkpoint=None):
        self.num_classes = num_classes
        self.num_test_samples = num_test_samples

        """
        # only correct when targets are 0 and 1
        for x in train_dataset.targets:
            assert x == 0 or x == 1
        w1 = sum(train_dataset.targets)/len(train_dataset.targets)
        self.class_weights = [1/(1-w1), 1/w1]
        """
        self.class_weights = None
        

        # Hyper Parameters
        self.batch_size = 128
        learning_rate = args.lr

        if args.forget_rate is None:
            if args.noise_type == "asymmetric":
                forget_rate = args.noise_rate / 2
            else:
                forget_rate = args.noise_rate
        else:
            forget_rate = args.forget_rate

        if args.dataset == 'nih' or args.noise_type == 'clean':
            self.noise_or_not = None
        else:
            self.noise_or_not = train_dataset.noise_or_not

        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        self.alpha_plan = [learning_rate] * args.n_epoch
        self.beta1_plan = [mom1] * args.n_epoch

        for i in range(args.epoch_decay_start, args.n_epoch):
            self.alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
            self.beta1_plan[i] = mom2

        # define drop rate schedule
        self.rate_schedule = np.ones(args.n_epoch) * forget_rate
        self.rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** args.exponent, args.num_gradual)

        self.device = device
        self.num_iter_per_epoch = args.num_iter_per_epoch
        self.print_freq = args.print_freq
        self.co_lambda = args.co_lambda
        self.n_epoch = args.n_epoch
        self.train_dataset = train_dataset

        if args.model_type == "cnn":
            self.model1 = CNN(input_channel=input_channel, n_outputs=num_classes)
            self.model2 = CNN(input_channel=input_channel, n_outputs=num_classes)
        elif args.model_type == "mlp":
            self.model1 = MLPNet()
            self.model2 = MLPNet()
        elif args.model_type == "resnet":
            # self.model1 = resnet34(num_classes=num_classes)
            # self.model2 = resnet34(num_classes=num_classes)
            self.model1 = Multihead_Resnet(num_classes=num_classes, device=device, head_elements=args.head_elements)
            self.model2 = Multihead_Resnet(num_classes=num_classes, device=device, head_elements=args.head_elements)

        if args.multi_gpu == "True":
            self.model1 = DataParallel(self.model1)
            self.model2 = DataParallel(self.model2)

        self.model1.to(device)
        print(self.model1.parameters)

        self.model2.to(device)
        print(self.model2.parameters)

        self.optimizer = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()),
                                          lr=learning_rate)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor= 0.8, patience=40, mode='min')

        self.loss_fn = loss_jocor


        self.adjust_lr = args.adjust_lr

        if start_checkpoint is not None:
            self.model1.load_state_dict(start_checkpoint['model1'])
            self.model2.load_state_dict(start_checkpoint['model2'])
            self.optimizer.load_state_dict(start_checkpoint['optimizer'])
            self.scheduler = start_checkpoint['scheduler']


    def evaluate_model(self, test_loader, model):
        model.eval()  # Change model to 'eval' mode.
        # correct = 0
        # total = 0
        # all_outputs = np.empty((0, self.num_classes), float)
        # all_labels = np.empty((0, ), int)

        all_outputs = np.empty((self.num_test_samples, self.num_classes), float)
        all_labels = np.empty((self.num_test_samples, self.num_classes), int)
        # all_outputs = np.empty((self.num_test_samples, ), float)
        # all_labels = np.empty((self.num_test_samples, ), int)

        cur_ind = 0

        for images, labels, _ in test_loader:
            images = Variable(images).to(self.device)
            logits = model(images)
            # outputs = F.softmax(logits, dim=1)  # for crossentropy loss
            outputs = torch.sigmoid(logits)  # for BCEloss

            # _, pred = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (pred.cpu() == labels).sum()

            # all_outputs = np.append(all_outputs, outputs.cpu().detach().numpy(), axis=0)
            # all_labels = np.append(all_labels, labels.cpu().detach().numpy())

            all_outputs[cur_ind:cur_ind+len(labels), :] = outputs.cpu().detach().numpy()
            all_labels[cur_ind:cur_ind+len(labels), :] = labels.cpu().detach().numpy()
            # all_outputs[cur_ind:cur_ind+len(labels)] = outputs.cpu().detach().numpy()[:,1]
            # all_labels[cur_ind:cur_ind+len(labels)] = labels.cpu().detach().numpy()
            cur_ind += len(labels)
        
        assert cur_ind == self.num_test_samples
        # acc = 100 * float(correct) / float(total)
        acc = 0.0
        auc = roc_auc_score(all_labels, all_outputs, average=None)  # alloutputs for multiLabel classification, alloutputs[:, 1] for binary
        return acc, auc


    # Evaluate the Model
    def evaluate(self, test_loader):
        print('Evaluating ...')

        acc1, auc1 = self.evaluate_model(test_loader, self.model1)
        acc2, auc2 = self.evaluate_model(test_loader, self.model2)

        return acc1, acc2, auc1, auc2

    """
    def roc_auc(self, test_dataset, args):
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=len(test_dataset),
                                                num_workers=args.num_workers,
                                                drop_last=True,
                                                shuffle=False)
        
        for images, labels, _ in test_loader:
            images = Variable(images).to(self.device)
            logits1 = self.model1(images)
            outputs1 = F.softmax(logits1, dim=1)

            outputs1 = outputs1.cpu().detach().numpy()
            print(outputs1.shape)
            assert len(outputs1[0]) == args.num_classes
            aucs = []
            for i in range(args.num_classes):
                output_i = outputs1[:, i]
                aucs.append(roc_auc_score(output_i, labels == i))

        return aucs
    """

    # Train the Model
    def train(self, train_loader, epoch):
        print('Training ...')
        self.model1.train()  # Change model to 'train' mode.
        self.model2.train()  # Change model to 'train' mode

        if self.adjust_lr == 1:
            self.adjust_learning_rate(self.optimizer, epoch)

        # train_total = 0
        # train_correct = 0
        # train_total2 = 0
        # train_correct2 = 0
        pure_ratio_1_list = []
        pure_ratio_2_list = []

        # last_loss

        for i, (images, labels, indexes) in enumerate(train_loader):
            ind = indexes.cpu().numpy().transpose()
            if i > self.num_iter_per_epoch:
                break

            images = Variable(images).to(self.device)
            labels = Variable(labels).to(self.device)

            # Forward + Backward + Optimize
            logits1 = self.model1(images)
            # prec1 = accuracy(logits1, labels, topk=(1,))
            # train_total += 1
            # train_correct += prec1

            logits2 = self.model2(images)
            # prec2 = accuracy(logits2, labels, topk=(1,))
            # train_total2 += 1
            # train_correct2 += prec2

            loss_1, loss_2, pure_ratio_1, pure_ratio_2 = self.loss_fn(logits1, logits2, labels, self.rate_schedule[epoch],
                                                                 ind, self.noise_or_not, self.co_lambda, self.class_weights)

            self.optimizer.zero_grad()
            loss_1.backward()
            self.optimizer.step()

            # pure_ratio_1_list.append(100 * pure_ratio_1)
            # pure_ratio_2_list.append(100 * pure_ratio_2)

            if (i + 1) % self.print_freq == 0:
                """
                print(
                    'Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss: %.4f, learning_rate: %.4f'
                    % (epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1, prec2,
                       loss_1.data.item(), self.optimizer.param_groups[0]['lr'])
                )"""
                """
                print(
                    'Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f, Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%'
                    % (epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1, prec2,
                       loss_1.data.item(), loss_2.data.item(), sum(pure_ratio_1_list) / len(pure_ratio_1_list), sum(pure_ratio_2_list) / len(pure_ratio_2_list)))
                """
                print(
                    'Epoch [%d/%d], Iter [%d/%d], Loss: %.4f, learning_rate: %.4f'
                    % (epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size,
                       loss_1.data.item(), self.optimizer.param_groups[0]['lr'])
                )
                if self.adjust_lr == 0:
                    self.scheduler.step(loss_1)

        # train_acc1 = float(train_correct) / float(train_total)
        # train_acc2 = float(train_correct2) / float(train_total2)
        train_acc1 = 0.0
        train_acc2 = 0.0
        return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1
