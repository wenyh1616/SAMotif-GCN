# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import jittor
import jittor.nn as nn

from net.utils.tgcn_med import MotifSTGraphical
from net.utils.graph_origin import Graph
import math
from net.utils.trans_skeleton_A_new import TransSkeleton
from net.utils.densenet_efficient_after import DenseBlock, Transition
from net.utils.non_local_embedded_gaussian_1DmeanV2 import NONLocalBlock1D

class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        vtdb_args (dict): The auguments for buiding the VTDB
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args, vtdb_args,
                 edge_importance_weighting, pyramid_pool, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        self._A = jittor.float32(jittor.Var(self.graph.A))
        # A.stop_grad()
        # self.register_buffer('A', A)
        # setattr(self,'A',A)
        # build networks
        spatial_kernel_size = self._A.size(0)

        temporal_kernel_size = [9, 9, 9]
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * self._A.size(1))

        self.motif_gcn_networks = nn.ModuleList([
            motif_gcn(in_channels, 64, kernel_size, 1, residual=False, **vtdb_args, **kwargs),
            motif_gcn(64, 64, kernel_size, 1, **vtdb_args, **kwargs),
            motif_gcn(64, 64, kernel_size, 1, **vtdb_args, **kwargs),
            motif_gcn(64, 64, kernel_size, 1, **vtdb_args, **kwargs),
            motif_gcn(64, 128, kernel_size, 2, **vtdb_args, **kwargs),
            motif_gcn(128, 128, kernel_size, 1, **vtdb_args, **kwargs),
            motif_gcn(128, 128, kernel_size, 1, **vtdb_args, **kwargs),
            motif_gcn(128, 256, kernel_size, 2, NL=True,**vtdb_args, **kwargs),
            motif_gcn(256, 256, kernel_size, 1, **vtdb_args, **kwargs),
            motif_gcn(256, 256, kernel_size, 1, **vtdb_args, **kwargs),
        ])



        self.intri_importance = nn.Parameter(jittor.ones([1, self._A.size(1), self._A.size(2)]))

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

        self.use_pyramid = False
        self.trans = TransSkeleton()


    def set_bn_momentum(self, momentum):
        for motif in self.motif_gcn_networks:
            motif.momentum = momentum

    def execute(self, x):


        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2)
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2)
        x = x.view(N * M, C, T, V)

        # forward
        for gcn in self.motif_gcn_networks:

            x_med = self.trans(x, self._A)
            x, _ = gcn(x, self._A, x_med * self.intri_importance)

        if self.use_pyramid:
            x = self.pool(x)
            x = x.view(N, M, -1, 5, 1, 1).mean(dim=1)
            x = x.mean(dim=2)
        else:
            # global pooling
            x =  nn.avg_pool2d(x, tuple(x.size()[2:]))
            x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x


class HalfSplit(nn.Module):
    def __init__(self, dim=0, first_half=True):
        super(HalfSplit, self).__init__()
        self.first_half = first_half
        self.dim = dim

    def execute(self, input):
        splits = jittor.chunk(input, 2, dim=self.dim)
        return splits[0] if self.first_half else splits[1]

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def execute(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).view(N, C, H, W)

class motif_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        interc (int, optional): control the number of channels in the input of VTDB. Default: 2
        gr (int, optional): control the growth rate of dense block in VTDB. Default: 4
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 NL=False,
                 interc=2,
                 gr=4,
                 dropout=0,
                 momentum=0.1,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2

        padding_s = (kernel_size[0][0] - 1) // 2
        padding_m = (kernel_size[0][1] - 1) // 2
        padding_l = (kernel_size[0][2] - 1) // 2


        self.momentum = momentum

        self.residual = residual
        if not residual:
            self.res = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.res = lambda x: x

        else:
            self.res = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels, momentum=momentum),
            )

        self.gcn = MotifSTGraphical(in_channels, out_channels, kernel_size[1])


        if not self.residual:
            inter_channels = out_channels
        else:
            inter_channels = out_channels // 2
            self.first_half = HalfSplit(dim=1, first_half=True)
            self.second_split = HalfSplit(dim=1, first_half=False)
            if stride == 1:
                self.res2 = lambda x: x

            else:
                self.res2 = nn.Sequential(
                    nn.Conv2d(
                        inter_channels,
                        inter_channels,
                        kernel_size=1,
                        stride=(stride, 1)),
                    nn.BatchNorm2d(inter_channels, momentum=momentum),
                )


        self.seq0 = nn.Sequential(nn.BatchNorm2d(out_channels, momentum=momentum),nn.ReLU())
        self.seq = nn.Sequential(nn.BatchNorm2d(inter_channels, momentum=momentum),nn.ReLU())
        num_layers = 3
        growth_rate = inter_channels // gr
        block = DenseBlock(
            num_layers=num_layers,
            num_input_features=inter_channels,
            growth_rate=growth_rate,
            kernel_size=[kernel_size[0][0], kernel_size[0][1], kernel_size[0][2]],
            stride=(1, 1),
            padding=[padding_s, padding_m, padding_l],
            drop_rate=dropout,
            efficient=False,
        )
        self.nl = NL
        if NL:
            self.NL = NONLocalBlock1D(in_channels=out_channels, inter_channels=None, sub_sample=False,
                                      bn_layer=True)
        self.tcn = nn.Sequential()

        self.tcn.add_module('denseblock', block)
        num_features = inter_channels + num_layers * growth_rate

        trans = Transition(num_input_features=num_features, num_output_features=inter_channels, kernel=1,
                           stride=stride, padding=0)
        self.tcn.add_module('transition', trans)

        self.relu = nn.ReLU()

    def execute(self, x, A, x_med):

        res = self.res(x)
        x, A = self.gcn(x, A, x_med)
        x = self.seq0(x)
        if self.nl:
            x = self.NL(x)
        if not self.residual:
            x = self.tcn(x) + res

        else:

            x2 = self.second_split(x)
            x2 = self.seq(x2)
            x2 = self.res2(x2)
            x1 = self.first_half(x)
            x1 = self.tcn(x1)
            x = jittor.concat([x1, x2], dim=1)

            x = x + res


        return self.relu(x), A


