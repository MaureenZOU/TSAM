# Based on https://github.com/avalonstrel/GatedConvolution_pytorch/
import torch

from base import BaseModel
from model.networks import (
    Generator,
    SNTemporalPatchGANDiscriminator
)

from model.vgg import Vgg16
from model.loss_module import ReconLoss, VGGLoss, StyleLoss, EdgeLoss, AdversarialLoss, ReconLoss_Mask

loss_nickname_to_module = {'loss_recon':ReconLoss, 'loss_masked_recon':ReconLoss_Mask, 'loss_vgg':VGGLoss, 'loss_style':StyleLoss, 'loss_edge':EdgeLoss}

class VideoInpaintingModel(BaseModel):
    def __init__(self, opts, nc_in=5, nc_out=3, d_s_args={}, d_t_args={}, losses=None):
        super().__init__()
        self.d_t_args = {
            "nf": 32,
            "use_sigmoid": True,
            "norm": 'SN'
        }  # default values
        for key, value in d_t_args.items():
            # overwrite default values if provided
            self.d_t_args[key] = value

        self.d_s_args = {
            "nf": 32,
            "use_sigmoid": True,
            "norm": 'SN'
        }  # default values
        for key, value in d_s_args.items():
            # overwrite default values if provided
            self.d_s_args[key] = value

        nf = opts['nf']
        norm = opts['norm']
        use_bias = opts['bias']

        # warning: if 2d convolution is used in generator, settings (e.g. stride,
        # kernal_size, padding) on the temporal axis will be discarded
        self.conv_by = opts['conv_by'] if 'conv_by' in opts else '3d'
        self.conv_type = opts['conv_type'] if 'conv_type' in opts else 'gated'
        self.flow_tsm = opts['flow_tsm']

        self.use_refine = opts['use_refine'] if 'use_refine' in opts else False
        use_skip_connection = opts.get('use_skip_connection', False)

        self.backbone = opts['backbone'] if 'backbone' in opts else 'unet'

        self.opts = opts

        ######################
        # Convolution layers #
        ######################
        self.generator = Generator(self.backbone,
            nc_in, nc_out, nf, use_bias, norm, self.conv_by, self.conv_type,
            use_refine=self.use_refine, use_skip_connection=use_skip_connection, use_flow_tsm=self.flow_tsm)

        #################
        # Discriminator #
        #################

        if 'spatial_discriminator' not in opts or opts['spatial_discriminator']:
            self.spatial_discriminator = SNTemporalPatchGANDiscriminator(
                nc_in=5, conv_type='2d', **self.d_s_args
            )
            self.advloss = AdversarialLoss()

        if 'temporal_discriminator' not in opts or opts['temporal_discriminator']:
            self.temporal_discriminator = SNTemporalPatchGANDiscriminator(
                nc_in=5, **self.d_t_args
            )
            self.advloss = AdversarialLoss()

        #######
        # Vgg #
        #######
        self.vgg = Vgg16(requires_grad=False)

        ########
        # Loss #
        ########
        self.losses = losses
        for key, value in losses.items():
            if value > 0:
                setattr(self, key, loss_nickname_to_module[key]())

    def forward(self, imgs, masks, guidances=None, flows=None, targets=None, outputs=None, validation=False, model='G'):
        if model == 'G_loss':
            masked_imgs = imgs * masks
            output = self.generator(masked_imgs, masks, guidances, flows)

            outputs = output['outputs']

            losses_value = {}
            # Loss is only computed during training
            if validation == False:
                outputs_feature = []
                targets_feature = []

                for frame_idx in range(targets.size(1)):
                    outputs_feature.append(self.vgg(outputs[:, frame_idx]))
                    targets_feature.append(self.vgg(targets[:, frame_idx]))

                output['vgg_outputs'] = outputs_feature
                output['vgg_targets'] = targets_feature

                for key, value in self.losses.items():
                    if value > 0:                
                        if 'mask' in key:
                            losses_value[key] = getattr(self, key)(targets, output, masks)
                        else:
                            losses_value[key] = getattr(self, key)(targets, output)
                output.pop('vgg_outputs')
                output.pop('vgg_targets')

                # compute gan-loss 
                if hasattr(self, 'temporal_discriminator'):
                    guidances = torch.full_like(masks, 0.) if guidances is None else guidances
                    input_imgs = torch.cat([outputs, masks, guidances], dim=2)
                    scores = self.temporal_discriminator(input_imgs)
                    gan_loss = self.advloss(scores, 1, False)
                    losses_value['gan_losses'] = gan_loss
            
            return losses_value, output
            # return output, output
        elif model == 'D_t_loss':
            # forward real
            guidances = torch.full_like(masks, 0.) if guidances is None else guidances
            losses_out = {}

            input_imgs = torch.cat([targets, masks, guidances], dim=2)
            scores = self.temporal_discriminator(input_imgs)
            loss_real = self.advloss(scores, 1, True)

            # forward fake
            input_imgs = torch.cat([outputs, masks, guidances], dim=2)
            scores = self.temporal_discriminator(input_imgs)
            loss_fake = self.advloss(scores, 0, True)

            losses_out['loss_real'] = loss_real
            losses_out['loss_fake'] = loss_fake

            return losses_out
        else:
            raise ValueError(f'forwarding model should be "G", "D_t", or "D_s", but got {model}')
        return output
