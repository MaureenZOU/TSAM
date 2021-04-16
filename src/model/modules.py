import torch
import torch.nn as nn

from model.blocks import (
    GatedConv, GatedDeconv,
    PartialConv, PartialDeconv,
    VanillaConv, VanillaDeconv
)

from model.tsm.module import interpolate


###########################
# Encoder/Decoder Modules #
###########################

class BaseModule(nn.Module):
    def __init__(self, conv_type, dconv_type=None):
        super().__init__()
        self.conv_type = conv_type
        if dconv_type == None:
            dconv_type = conv_type

        if conv_type == 'gated':
            self.ConvBlock = GatedConv
        elif conv_type == 'partial':
            self.ConvBlock = PartialConv
        elif conv_type == 'vanilla':
            self.ConvBlock = VanillaConv

        if dconv_type == 'gated':
            self.DeconvBlock = GatedDeconv
        elif dconv_type == 'partial':
            self.DeconvBlock = PartialDeconv
        elif dconv_type == 'vanilla':
            self.DeconvBlock = VanillaDeconv



class DownSampleModule(BaseModule):
    def __init__(self, nc_in, nf, use_bias, norm, conv_by, conv_type):
        super().__init__(conv_type)
        self.conv1 = self.ConvBlock(
            nc_in, nf * 1, kernel_size=(3, 5, 5), stride=1,
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)

        # Downsample 1
        self.conv2 = self.ConvBlock(
            nf * 1, nf * 2, kernel_size=(3, 4, 4), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv3 = self.ConvBlock(
            nf * 2, nf * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        # Downsample 2
        self.conv4 = self.ConvBlock(
            nf * 2, nf * 4, kernel_size=(3, 4, 4), stride=(1, 2, 2),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv5 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv6 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)

        # Dilated Convolutions
        self.dilated_conv1 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(1, 2, 2))
        self.dilated_conv2 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(1, 4, 4))
        self.dilated_conv3 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(1, 8, 8))
        self.dilated_conv4 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(1, 16, 16))
        self.conv7 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv8 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)

    def forward(self, inp):
        c1 = self.conv1(inp)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)

        a1 = self.dilated_conv1(c6)
        a2 = self.dilated_conv2(a1)
        a3 = self.dilated_conv3(a2)
        a4 = self.dilated_conv4(a3)

        c7 = self.conv7(a4)
        c8 = self.conv8(c7)
        return c8, c4, c2  # For skip connection


class AttentionDownSampleModule(DownSampleModule):
    def __init__(self, nc_in, nf, use_bias, norm, conv_by, conv_type):
        super().__init__(nc_in, nf, use_bias, norm, conv_by, conv_type)


class UpSampleModule(BaseModule):
    def __init__(self, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type,
                 use_skip_connection=False):
        super().__init__(conv_type)
        # Upsample 1
        self.deconv1 = self.DeconvBlock(
            nc_in * 2 if use_skip_connection else nc_in,
            nf * 2, kernel_size=(3, 3, 3), stride=1, padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv9 = self.ConvBlock(
            nf * 2, nf * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        # Upsample 2
        self.deconv2 = self.DeconvBlock(
            nf * 4 if use_skip_connection else nf * 2,
            nf * 1, kernel_size=(3, 3, 3), stride=1, padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv10 = self.ConvBlock(
            nf * 1, nf // 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv11 = self.ConvBlock(
            nf // 2, nc_out, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=None, activation=None, conv_by=conv_by)
        self.use_skip_connection = use_skip_connection

    def concat_feature(self, ca, cb):
        if self.conv_type == 'partial':
            ca_feature, ca_mask = ca
            cb_feature, cb_mask = cb
            feature_cat = torch.cat((ca_feature, cb_feature), 1)
            # leave only the later mask
            return feature_cat, ca_mask
        else:
            return torch.cat((ca, cb), 1)

    def forward(self, inp):
        c8, c4, c2 = inp
        if self.use_skip_connection:
            d1 = self.deconv1(self.concat_feature(c8, c4))
            c9 = self.conv9(d1)
            d2 = self.deconv2(self.concat_feature(c9, c2))
        else:
            d1 = self.deconv1(c8)
            c9 = self.conv9(d1)
            d2 = self.deconv2(c9)

        c10 = self.conv10(d2)
        c11 = self.conv11(c10)
        return c11


class UpSampleResNetSkip(BaseModule):
    def __init__(self, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type,
                 use_skip_connection=False):
        super().__init__(conv_type)
        assert False, "UpSampleResNetSkip has worse performance than UpSampleResNetSkipGated"
        # Upsample 1
        self.conv_c2 = self.ConvBlock(512, 256, kernel_size=(3, 1, 1), stride=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv_c4 = self.ConvBlock(1024, 256, kernel_size=(3, 1, 1), stride=1,
            bias=use_bias, norm=norm, conv_by=conv_by)

        self.deconv1 = self.DeconvBlock(2048, 256, kernel_size=(3, 1, 1), stride=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv9 = self.ConvBlock(
            256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)

        # Upsample 2
        self.deconv2 = self.DeconvBlock(
            256,
            256, kernel_size=(3, 1, 1), stride=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv10 = self.ConvBlock(
            256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)

        # Upsample 3
        self.deconv3 = self.DeconvBlock(
            256,
            256, kernel_size=(3, 1, 1), stride=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv11 = self.ConvBlock(
            256, nc_out, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=None, activation=None, conv_by=conv_by)

        for name, module in self.named_modules():
            if isinstance(module, (VanillaConv)):
                nn.init.kaiming_uniform_(module.featureConv.layer.weight, a=1)
                nn.init.constant_(module.featureConv.layer.bias, 0)
            elif isinstance(module, (VanillaDeconv)):
                nn.init.kaiming_uniform_(module.conv.featureConv.layer.weight, a=1)
                nn.init.constant_(module.conv.featureConv.layer.bias, 0)

    def forward(self, inp):
        c1, c2, c4, c8 = inp

        c2 = self.conv_c2(c2)
        c4 = self.conv_c4(c4)

        c4 = interpolate(c4, scale_factor=2, mode="nearest")
        c2 = interpolate(c2, scale_factor=4, mode="nearest")
        c1 = interpolate(c1, scale_factor=4, mode="nearest")

        d1 = self.deconv1(c8)
        d1 = d1 + c4
        c9 = self.conv9(d1)
        d2 = self.deconv2(c9)
        d2 = d2 + c2
        c10 = self.conv10(d2)
        d3 = self.deconv3(c10)
        d3 = d3 + c1
        c11 = self.conv11(d3)
        return c11

class UpSampleResNetSkipGated(BaseModule):
    def __init__(self, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type,
                 use_skip_connection=False, use_flow_tsm=False):
        super().__init__(conv_type, 'vanilla')
        # Upsample 1
        self.conv_c2 = self.ConvBlock(512, 256, kernel_size=(3, 1, 1), stride=1,
            bias=use_bias, norm=norm, conv_by=conv_by, use_flow_tsm=use_flow_tsm)
        self.conv_c4 = self.ConvBlock(1024, 256, kernel_size=(3, 1, 1), stride=1,
            bias=use_bias, norm=norm, conv_by=conv_by, use_flow_tsm=use_flow_tsm)

        self.deconv1 = self.DeconvBlock(2048, 256, kernel_size=(3, 1, 1), stride=1,
            bias=use_bias, norm=norm, conv_by="2d", use_flow_tsm=False)
        self.conv9 = self.ConvBlock(
            256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by, use_flow_tsm=use_flow_tsm)

        # Upsample 2
        self.deconv2 = self.DeconvBlock(
            256,
            256, kernel_size=(3, 1, 1), stride=1,
            bias=use_bias, norm=norm, conv_by="2d", use_flow_tsm=False)
        self.conv10 = self.ConvBlock(
            256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by, use_flow_tsm=use_flow_tsm)

        # Upsample 3
        self.deconv3 = self.DeconvBlock(
            256,
            256, kernel_size=(3, 1, 1), stride=1,
            bias=use_bias, norm=norm, conv_by="2d", use_flow_tsm=False)
        self.conv11 = self.ConvBlock(
            256, nc_out, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=None, activation=None, conv_by=conv_by, use_flow_tsm=use_flow_tsm)

        for name, module in self.named_modules():
            if isinstance(module, (GatedConv)):
                nn.init.kaiming_uniform_(module.featureConv.layer.weight, a=1)
                nn.init.constant_(module.featureConv.layer.bias, 0)
                nn.init.kaiming_uniform_(module.gatingConv.layer.weight, a=1)
                nn.init.constant_(module.gatingConv.layer.bias, 0)
            elif isinstance(module, (VanillaDeconv)):
                nn.init.kaiming_uniform_(module.conv.featureConv.layer.weight, a=1)
                nn.init.constant_(module.conv.featureConv.layer.bias, 0)
        
    def forward(self, inp, flows=None):
        c1, c2, c4, c8 = inp

        c2 = self.conv_c2(c2, flows)
        c4 = self.conv_c4(c4, flows)

        c4 = interpolate(c4, scale_factor=2, mode="nearest")
        c2 = interpolate(c2, scale_factor=4, mode="nearest")
        c1 = interpolate(c1, scale_factor=4, mode="nearest")

        d1 = self.deconv1(c8, flows)
        d1 = d1 + c4
        c9 = self.conv9(d1, flows)
        d2 = self.deconv2(c9, flows)
        d2 = d2 + c2
        c10 = self.conv10(d2, flows)
        d3 = self.deconv3(c10, flows)
        d3 = d3 + c1
        c11 = self.conv11(d3, flows)
        return c11