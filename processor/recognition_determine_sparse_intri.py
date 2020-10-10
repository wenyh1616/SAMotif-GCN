#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
from torchlight import str2bool

from .processor_determine_sparse_intri import Processor
#from net.utils.profile import *
# from torchvision.models import resnet152
# from thop import profile
# model = resnet50()
# input = torch.randn(1, 3, 224, 224)
# flops, params = profile(model, inputs=(input,))
# print("FLOPs", flops/(1000 * 1000 * 1000))
# print("params", params/(1000 * 1000))
# print("resnet")
from ptflops import get_model_complexity_info
# net = resnet152()
# flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
# print("")
# from torchstat import stat
# model = resnet152()
# result = stat(model, (3, 224, 224))
# 0
# Wen add:
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        # m.weight.data.fill_(0.01)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        # m.weight.data.fill_(0)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
def l1_loss(x):
    return torch.abs(x).sum()

class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        # flops, params = get_model_complexity_info(self.model, (3, 300, 18, 2), as_strings=True, print_per_layer_stat=True)
        # flops, params = get_model_complexity_info(self.model, (3, 300, 25, 2), as_strings=True,
        #                                           print_per_layer_stat=True)
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            lr = np.power(self.arg.d_model, -0.5) * np.min([
                np.power(self.meta_info['iter'] + 1, -0.5),
                np.power(self.arg.n_warmup_steps, -1.5) * (self.meta_info['iter'] + 1)])

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log(f'\tTop{k}: {100 * accuracy:.2f}%')


    def train(self):

        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for batch_idx, (data, label) in enumerate(loader):

            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)


            # forward
            output = self.model(data)
            to_regularize = []
            # to_regularize2 = []
            intri_importance = self.model.module.intri_importance
            # for param in edge_importance:
            #     param_use = param.view(-1)
            #     to_regularize.append(param_use)
            # A = self.model.module.graph.A.shape
            # A_use = torch.ones(1, A[1], A[2])

            for param in intri_importance:
                param_use = param.view(-1)
                to_regularize.append(param_use)
           # test =
            l1 = self.arg.lamda_r * l1_loss(torch.cat(to_regularize))
            loss = self.loss(output, label) + l1

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = f'{self.lr:.6f}'
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in loader:
            
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model(data)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:

                to_regularize = []
                # edge_importance = self.model.module.edge_importance
                intri_importance = self.model.module.intri_importance
                # for param in edge_importance:
                #     param_use = param.view(-1)
                #     to_regularize.append(param_use)
                for param in intri_importance:
                    param_use = param[-1].view(-1)
                    to_regularize.append(param_use)
                test =  l1_loss(torch.cat(to_regularize))
                l1 = self.arg.lamda_r

                loss = self.loss(output, label) + l1
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss']= np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--lamda_r', type=float, default=0.00008, help='initial learning rate')
        parser.add_argument('--lamda_m', type=float, default=0.5, help='initial learning rate')
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='Adam', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--n_warmup_steps', type=int, default=8000, help='warm up steps')
        parser.add_argument('--d_model', type=int, default=300, help='model size')

        # endregion yapf: enable

        return parser
