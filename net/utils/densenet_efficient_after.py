# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.checkpoint as cp
import jittor.nn as nn
import jittor

def _bn_function_factory(norm, relu, conv):

    def bn_function(*inputs):
        # concated_features = torch.cat(inputs, 1)
        concated_features = jittor.concat(inputs, 1)
        bottleneck_output = relu(norm(conv(concated_features)))
        return bottleneck_output

    return bn_function


class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, kernel_size, stride, padding, drop_rate, efficient=False):
        super(DenseLayer, self).__init__()
        # self.add_module('norm1', nn.BatchNorm2d(growth_rate, momentum=0.1)),
        # self.add_module('relu1', nn.ReLU(inplace=True)),
        # self.add_module('conv1', nn.Conv2d(num_input_features, growth_rate, kernel_size=kernel_size, stride=stride,
        #                                    padding=padding, bias=False)),
        # self.norm1 = nn.BatchNorm2d(growth_rate, momentum=0.1)
        # self.relu1 = nn.ReLU
        # self.conv1 = nn.Conv2d(num_input_features, growth_rate, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        setattr(self,'norm1',nn.BatchNorm2d(growth_rate, momentum=0.1))
        setattr(self,'relu1',nn.ReLU())
        setattr(self,'conv1',nn.Conv2d(num_input_features, growth_rate, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        self.drop_rate = drop_rate
        self.efficient = efficient

    def execute(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        # if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
        #     bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        # else:
        #     bottleneck_output = bn_function(*prev_features)
        bottleneck_output = bn_function(*prev_features)
        if self.drop_rate > 0:
            # bottleneck_output = F.dropout(bottleneck_output, p=self.drop_rate, training=self.training)
            bottleneck_output = nn.dropout(bottleneck_output, p=self.drop_rate, is_train=self.is_training())
        return bottleneck_output


class Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, kernel, stride, padding):
        super(Transition, self).__init__()

        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=(kernel, 1), stride=(stride, 1), padding=(padding,0), bias=False))
        self.add_module('norm', nn.BatchNorm2d(num_output_features))
        # self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=(kernel, 1), stride=(stride, 1), padding=(padding,0), bias=False)
        # self.norm = nn.BatchNorm2d(num_output_features)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, kernel_size, stride, padding, drop_rate, efficient=False):
        super(DenseBlock, self).__init__()
        self.denselayer = nn.ModuleList()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                kernel_size=(kernel_size[i], 1),
                stride=stride,
                padding=(padding[i], 0),
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.denselayer.add_module('denselayer%d' % (i + 1), layer)
            # self.add_module('denselayer%d' % (i + 1), layer)
            # setattr(self,'denselayer%d' % (i + 1),layer)

    def execute(self, init_features):
        features = [init_features]
        for name, layer in self.denselayer.items():
            # print(type(layer))
            new_features = layer(*features)
            features.append(new_features)
        return jittor.concat(features, 1)
