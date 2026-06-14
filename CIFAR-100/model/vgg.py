"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F' : [84, 84, 'M', 84*2, 84*2, 'M', 84*4, 84*4, 84*4, 84*4, 'M', 84*8, 84*8, 84*8, 84*8, 'M', 84*8, 84*8, 84*8, 84*8, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=14):
        super().__init__()
        self.features = features
        # print(self.features)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.layer_4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.layer_5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.layer_6 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.layer_7 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.layer_8 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        self.bn2 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(32)
        self.bn8 = nn.BatchNorm2d(16)

        self.classifier = nn.Sequential(
            nn.Linear(802816, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_class)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):

        # output = self.features(x)
        out_1 = self.layer_1(x)
        out_1 = self.bn1(out_1)
        out_1 = self.relu(out_1)

        out_2 = self.layer_2(out_1)
        out_2 = self.bn2(out_2)
        out_2 = self.relu(out_2)


        out_1 = out_2 - out_1

        out_3 = self.layer_5(out_1)
        out_3 = self.bn5(out_3)
        out_3 = self.relu(out_3)
        """
        out_4 = self.layer_4(out_3)
        out_4 = self.bn4(out_4)
        out_4 = self.relu(out_4)
        """
        out_3 = self.layer_6(out_3)
        out_3 = self.bn6(out_3)
        out_3 = self.relu(out_3)

        out_3 = self.layer_7(out_3)
        out_3 = self.bn7(out_3)
        out_3 = self.relu(out_3)

        out_3 = self.layer_8(out_3)
        out_3 = self.bn8(out_3)
        out_3 = self.relu(out_3)

        # print(output.size())
        output = out_3.view(out_3.size()[0], -1)

        # print(output.size())
        output = self.classifier(output)


        # print(output.size())
        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))

def vgg16_bn_84():
    return VGG(make_layers(cfg['F'], batch_norm=True))


