"""Super-Resolution model."""
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.models import vgg19
import torchvision
from math import sqrt

class Model_4x(nn.Module):
    def __init__(self, upscale_factor=4, img_channels=3, out_channels = 3, feat_size=64,
                 nof_blocks=10, activation='relu'):
        super().__init__()

        self.upscale_factor = upscale_factor
        self.nof_blocks = nof_blocks
        self.out_channels = out_channels
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.conv_in = nn.Conv2d(img_channels, feat_size, 3, padding=1)
        self.instance_normalization = nn.InstanceNorm2d(3, track_running_stats=True)
        self.conv_blocks = nn.ModuleList()
        for _ in range(self.nof_blocks):
            self.conv_blocks.append(
                nn.ModuleList([nn.Conv2d(feat_size, feat_size, 3, padding=1),
                               nn.Conv2d(feat_size, feat_size, 3, padding=1)]))

        self.conv_up_new = nn.Conv2d(feat_size, 4*feat_size, 3, padding=1)
        self.nn_up1 = nn.PixelShuffle(2)
        self.conv_up1 = nn.Conv2d(feat_size, 4*feat_size, 3, padding=1)
        self.nn_up2 = nn.PixelShuffle(2)
        self.conv_up2 = nn.Conv2d(feat_size, feat_size, 3, padding=1)
        self.conv_out = nn.Conv2d(feat_size, self.out_channels, 3, padding=1)
        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv_in(x))
        for block in self.conv_blocks:
            res = self.relu(block[0](x))
            res = block[1](res)
            x = x + res
        x = self.conv_up_new(x)
        x = self.relu(x)
        x = self.nn_up1(x)
        x = self.conv_up1(x)
        x = self.relu(x)
        x = self.nn_up2(x)
        x = self.conv_up2(x)
        x = self.relu(x)
        return self.conv_out(x)

    def _initialize_weights(self):
        init.xavier_uniform_(self.conv_in.weight, gain=sqrt(2))
        for i,block in enumerate(self.conv_blocks):
            init.xavier_uniform_(block[0].weight, gain=sqrt(2))
            init.xavier_uniform_(block[1].weight, gain=sqrt(2))
        init.xavier_uniform_(self.conv_up_new.weight, gain=sqrt(2))
        init.xavier_uniform_(self.conv_up1.weight, gain=sqrt(2))
        init.xavier_uniform_(self.conv_up2.weight, gain=sqrt(2))
        init.xavier_uniform_(self.conv_out.weight, gain=sqrt(2))

class Model_8x(nn.Module):

    def __init__(self, upscale_factor=8, img_channels=3, out_channels = 3, feat_size=64,
                 nof_blocks=10, activation='relu'):
        super().__init__()

        self.upscale_factor = upscale_factor
        self.nof_blocks = nof_blocks
        self.out_channels = out_channels
        self.relu = nn.ReLU()
        self.conv_in = nn.Conv2d(img_channels, feat_size, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(feat_size)
        self.conv_blocks = nn.ModuleList()
        for _ in range(self.nof_blocks):
            self.conv_blocks.append(
                nn.ModuleList([nn.Conv2d(feat_size, feat_size, 3, padding=1),
                               nn.Conv2d(feat_size, feat_size, 3, padding=1)]))
        self.conv_up_new = nn.Conv2d(feat_size, 4*feat_size, 3, padding=1)
        self.nn_up1 = nn.PixelShuffle(2)
        self.conv_up1 = nn.Conv2d(feat_size, 4*feat_size, 3, padding=1)
        self.nn_up2 = nn.PixelShuffle(2)
        self.conv_up2 = nn.Conv2d(feat_size, feat_size*4, 3, padding=1)
        self.nn_up3 = nn.PixelShuffle(2)
        self.conv_up3 = nn.Conv2d(feat_size, feat_size, 3, padding=1)
        self.conv_out = nn.Conv2d(feat_size, self.out_channels, 3, padding=1)
        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv_in(x))
        for block in self.conv_blocks:
            res = self.relu(block[0](x))
            res = block[1](res)
            x = x + res
        x = self.relu(self.conv_up_new(x))
        x = self.nn_up1(x)
        x = self.conv_up1(x)
        x = self.relu(x)
        x = self.nn_up2(x)
        x = self.conv_up2(x)
        x = self.relu(x)
        x = self.nn_up3(x)
        x = self.conv_up3(x)
        x = self.relu(x)
        return self.conv_out(x)

    def _initialize_weights(self):
        for i,block in enumerate(self.conv_blocks):
            init.xavier_uniform_(block[0].weight, gain=sqrt(2))
            init.xavier_uniform_(block[1].weight, gain=sqrt(2))
        init.xavier_uniform_(self.conv_up_new.weight, gain=sqrt(2))
        init.xavier_uniform_(self.conv_up1.weight, gain=sqrt(2))
        init.xavier_uniform_(self.conv_up2.weight, gain=sqrt(2))
        init.xavier_uniform_(self.conv_up3.weight, gain=sqrt(2))
        init.xavier_uniform_(self.conv_out.weight, gain=sqrt(2))

class VGG(nn.Module):
    'Pretrained VGG-19 model features.'
    def __init__(self, layers=(0), replace_pooling = False):
        super(VGG, self).__init__()
        self.layers = layers
        self.instance_normalization = nn.InstanceNorm2d(128)
        self.relu = nn.ReLU()
        self.model = vgg19(pretrained=True).features
        # Changing Max Pooling to Average Pooling
        if replace_pooling:
            self.model._modules['4'] = nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['9'] = nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['18'] =nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['27'] =nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['36'] = nn.AvgPool2d((2,2), (2,2), (1,1))
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for name, layer in enumerate(self.model):
            x = layer(x)
            if name in self.layers:
                features.append(x)
                if len(features) == len(self.layers):
                    break
        return features
