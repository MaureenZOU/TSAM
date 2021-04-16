import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(torch.nn.Conv2d):
    def forward(self, x, flows):
        B,C,L,H,W = x.shape
        x = x.transpose(1,2).reshape(B*L,C,H,W)
        x = super(Conv2d, self).forward(x)
        _,C,H,W = x.shape
        return x.reshape(B,L,C,H,W).transpose(1,2)


class BatchNorm2d(torch.nn.BatchNorm2d):
    def forward(self, x, flows):
        B,C,L,H,W = x.shape
        x = x.transpose(1,2).reshape(B*L,C,H,W)
        x = super(BatchNorm2d, self).forward(x)
        return x.reshape(B,L,C,H,W).transpose(1,2)


class Relu(torch.nn.ReLU):
    def forward(self, x, flows):
        B,C,L,H,W = x.shape
        x = x.transpose(1,2).reshape(B*L,C,H,W)
        x = super(Relu, self).forward(x)
        return x.reshape(B,L,C,H,W).transpose(1,2)


class MaxPool2d(torch.nn.MaxPool2d):
    def forward(self, x, flows):
        B,C,L,H,W = x.shape
        x = x.transpose(1,2).reshape(B*L,C,H,W)
        x = super(MaxPool2d, self).forward(x)
        _,C,H,W = x.shape
        return x.reshape(B,L,C,H,W).transpose(1,2)

class Sequential(torch.nn.Sequential):
    def forward(self, x, flows):
        for module in self:
            x = module(x, flows)
        return x


def interpolate(x, scale_factor=2, mode='nearest'):
    B,C,L,H,W = x.shape
    x = x.transpose(1,2).reshape(B*L,C,H,W)
    x = F.interpolate(x, scale_factor=scale_factor, mode=mode)
    _,C,H,W = x.shape
    return x.reshape(B,L,C,H,W).transpose(1,2)


def conv_with_kaiming_uniform(use_gn=False, use_relu=False):
    def make_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            bias=False if use_gn else True
        )
        # Caffe2 implementation uses XavierFill, which in fact
        # corresponds to kaiming_uniform_ in PyTorch
        nn.init.kaiming_uniform_(conv.weight, a=1)
        if not use_gn:
            nn.init.constant_(conv.bias, 0)
        module = [conv,]
        if use_gn:
            module.append(group_norm(out_channels))
        if use_relu:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv