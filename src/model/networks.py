import functools
import torch
import torch.nn as nn

from model.modules import UpSampleModule, DownSampleModule, AttentionDownSampleModule, BaseModule, UpSampleResNetSkip, UpSampleResNetSkipGated


##################
# Generators #
##################

class CoarseNet(nn.Module):

    def __init__(self, backbone, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection: bool = False, use_flow_tsm=False):
        super().__init__()
        self.backbone = backbone
        self.conv_type = conv_type

        if self.backbone == 'unet':
            self.downsample_module = DownSampleModule(
                nc_in, nf, use_bias, norm, conv_by, conv_type)
            self.upsample_module = UpSampleModule(
                nf * 4, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection)

        elif self.backbone == 'resnet' and conv_by == '2dtsm' and conv_type == 'vanilla':
            from model.tsm.models import TSN
            self.downsample_module = TSN(400, 3, 'RGB',
                base_model='resnet50',
                consensus_type='avg',
                dropout=0.5,
                img_feature_dim=256,
                partial_bn=False,
                pretrain='none',
                is_shift=True, shift_div=8, shift_place='blockres',
                fc_lr5=True,
                temporal_pool=False,
                non_local=False,
                replace_stride_with_dilation=[False, True, True],
                gated=False,
                use_flow_tsm=use_flow_tsm)
            self.upsample_module = UpSampleResNetSkip(
                nf * 4, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection, use_flow_tsm)

        elif self.backbone == 'resnet' and conv_by == '2dtsm' and conv_type == 'gated':
            from model.tsm.models import TSN
            self.downsample_module = TSN(400, 3, 'RGB',
                base_model='resnet50',
                consensus_type='avg',
                dropout=0.5,
                img_feature_dim=256,
                partial_bn=False,
                pretrain='none',
                is_shift=True, shift_div=8, shift_place='blockres',
                fc_lr5=True,
                temporal_pool=False,
                non_local=False,
                replace_stride_with_dilation=[False, True, True],
                gated=True,
                use_flow_tsm=use_flow_tsm)
            self.upsample_module = UpSampleResNetSkipGated(
                nf * 4, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection, use_flow_tsm)

        else:
            assert False, "the backbone type is not implemented"
            
    def preprocess(self, masked_imgs, masks, guidances, flows):
        # B, L, C, H, W = masked.shape
        masked_imgs = masked_imgs.transpose(1, 2)
        masks = masks.transpose(1, 2)
        flows = [flow.transpose(1, 2) for flow in flows]

        if self.conv_type == 'partial':
            if guidances is not None:
                raise NotImplementedError('Partial convolution does not support guidance')
            # the input and output of partial convolution are both tuple (imgs, mask)
            inp = (masked_imgs, masks)
        elif self.conv_type == 'gated' or self.conv_type == 'vanilla':
            guidances = torch.full_like(masks, 0.) if guidances is None else guidances.transpose(1, 2)
            inp = torch.cat([masked_imgs, masks, guidances], dim=1)
        else:
            raise NotImplementedError(f"{self.conv_type} not implemented")

        return inp, flows 

    def postprocess(self, masked_imgs, masks, c11):
        if self.conv_type == 'partial':
            inpainted = c11[0].transpose(1, 2) * (1 - masks)
        else:
            inpainted = c11.transpose(1, 2) * (1 - masks)

        out = inpainted + masked_imgs
        return out

    def forward(self, masked_imgs, masks, guidances=None, flows=None):
        # B, L, C, H, W = masked.shape
        inp, flows = self.preprocess(masked_imgs, masks, guidances, flows)

        encoded_features = self.downsample_module(inp, flows)
        c11 = self.upsample_module(encoded_features, flows)

        out = self.postprocess(masked_imgs, masks, c11)

        return out


class RefineNet(CoarseNet):
    def __init__(self, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection: bool = False):
        super().__init__(nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection)
        self.upsample_module = UpSampleModule(
            nf * 16, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection)
        self.attention_downsample_module = AttentionDownSampleModule(
            nc_in, nf, use_bias, norm, conv_by, conv_type)

    def forward(self, coarse_output, masks, guidances=None):
        inp = self.preprocess(coarse_output, masks, guidances)

        encoded_features = self.downsample_module(inp)

        attention_features, offset_flow = self.attention_downsample_module(inp)

        deconv_inp = torch.cat([encoded_features, attention_features], dim=2)

        c11 = self.upsample_module(deconv_inp)

        out = self.postprocess(coarse_output, masks, c11)
        return out, offset_flow


class Generator(nn.Module):
    def __init__(
        self, backbone, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, use_refine=False,
        use_skip_connection: bool = False, use_flow_tsm=False
    ):
        super().__init__()
        self.coarse_net = CoarseNet(
            backbone, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection, use_flow_tsm
        )
        self.use_refine = use_refine
        if self.use_refine:
            self.refine_net = RefineNet(
                nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, use_skip_connection
            )

    def forward(self, masked_imgs, masks, guidances=None, flows=None):
        coarse_outputs = self.coarse_net(masked_imgs, masks, guidances, flows)
        if self.use_refine:
            refined_outputs, offset_flows = self.refine_net(coarse_outputs, masks, guidances)
            return {
                "outputs": refined_outputs,
                "offset_flows": offset_flows,
                "coarse_outputs": coarse_outputs
            }
        else:
            return {"outputs": coarse_outputs}


##################
# Discriminators #
##################

class SNTemporalPatchGANDiscriminator(BaseModule):
    def __init__(
        self, nc_in, nf=64, norm='SN', use_sigmoid=True, use_bias=True, conv_type='vanilla',
        conv_by='3d'
    ):
        super().__init__(conv_type)
        use_bias = use_bias
        self.use_sigmoid = use_sigmoid

        ######################
        # Convolution layers #
        ######################
        self.conv1 = self.ConvBlock(
            nc_in, nf * 1, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv2 = self.ConvBlock(
            nf * 1, nf * 2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv3 = self.ConvBlock(
            nf * 2, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv4 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv5 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv6 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=None, activation=None,
            conv_by=conv_by
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        # B, L, C, H, W = xs.shape
        xs_t = torch.transpose(xs, 1, 2)
        c1 = self.conv1(xs_t)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        if self.use_sigmoid:
            c6 = torch.sigmoid(c6)
        out = torch.transpose(c6, 1, 2)
        return out


# Based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class TemporalPatchGANDiscriminator(nn.Module):
    def __init__(self, input_nc, output_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = (3, 4, 4)
        padw = (1, 1, 1)
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=(1, 2, 2), padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=(1, 2, 2), padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, output_nc, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)