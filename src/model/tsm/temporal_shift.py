# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from model.tsm.module import Sequential
from utils.visual_helper import resize_flow, flow_to_grid


class GatedTemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(GatedTemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace

        # define gated convolution layer
        self.gating_conv = copy.deepcopy(net)
        self.sigmoid = nn.Sigmoid()

        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x, flows):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        x = self.sigmoid(self.gating_conv(x, flows)) * self.net(x, flows)

        return x

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        B,C,L,H,W = x.shape
        x = x.transpose(1,2)

        fold = C // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.transpose(1,2)


class FlowGatedTemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(FlowGatedTemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace

        # define gated convolution layer
        self.gating_conv = copy.deepcopy(net)
        self.sigmoid = nn.Sigmoid()

        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x, flows):
        x = self.shift(x, self.n_segment, flows, fold_div=self.fold_div, inplace=self.inplace)
        x = self.sigmoid(self.gating_conv(x, flows)) * self.net(x, flows)
        return x

    @staticmethod
    def shift(x, n_segment, flows, fold_div=8, inplace=False):
        B,C,L,H,W = x.shape
        x = x.transpose(1,2)

        left_flow, right_flow, left_flowmask, right_flowmask = flows
        _,_,_,h,w = left_flow.shape

        left_flow = left_flow.transpose(1,2).reshape(B*(L-1),2,h,w)
        right_flow = right_flow.transpose(1,2).reshape(B*(L-1),2,h,w)
        left_flowmask = left_flowmask.transpose(1,2).reshape(B*(L-1),1,h,w)
        right_flowmask = right_flowmask.transpose(1,2).reshape(B*(L-1),1,h,w)        

        fold = C // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            left_flow = flow_to_grid(resize_flow(left_flow, (H,W))).permute(0,2,3,1)
            right_flow = flow_to_grid(resize_flow(right_flow, (H,W))).permute(0,2,3,1)

            left_flowmask = F.interpolate(left_flowmask, (H,W), mode='nearest').reshape(B,(L-1),1,H,W)
            right_flowmask = F.interpolate(right_flowmask, (H,W), mode='nearest').reshape(B,(L-1),1,H,W)

            left_feature = (F.grid_sample(x[:, 1:, :fold].reshape(B*(L-1),fold,H,W), left_flow).reshape(B,L-1,fold,H,W))
            right_feature = F.grid_sample(x[:, :-1, fold: 2 * fold].reshape(B*(L-1),fold,H,W), right_flow).reshape(B,L-1,fold,H,W)
            left_flowmask = torch.clamp(left_flowmask + (torch.sum(left_feature, dim=2, keepdim=True)==0).float(), 0, 1).int().float()
            right_flowmask = torch.clamp(right_flowmask + (torch.sum(left_feature, dim=2, keepdim=True)==0).float(), 0, 1).int().float()

            out[:, :-1, :fold] = left_feature*(1-left_flowmask) + left_flowmask*x[:, :-1, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = right_feature*(1-right_flowmask) + right_flowmask*x[:, 1:, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

            # DEBUG
            # import matplotlib.pyplot as plt
            # import matplotlib
            # matplotlib.use('Agg')

            # map_ = torch.norm(x[0:1,1, :fold], dim=1)[0].cpu().detach().numpy()
            # plt.matshow(map_)
            # plt.axis('off')
            # plt.savefig('before_shift_left.png', bbox_inches='tight', pad_inches = 0)
            # plt.clf()
            # plt.close()

            # map_ = torch.norm(out[0:1,0, :fold], dim=1)[0].cpu().detach().numpy()
            # plt.matshow(map_)
            # plt.axis('off')
            # plt.savefig('shift_left.png', bbox_inches='tight', pad_inches = 0)
            # plt.clf()
            # plt.close()

            # map_ = ((torch.norm(out[0:1,0, :fold], dim=1)[0].cpu().detach())==0).float().numpy()
            # plt.matshow(map_)
            # plt.axis('off')
            # plt.savefig('shift_left_mask.png', bbox_inches='tight', pad_inches = 0)
            # plt.clf()
            # plt.close()


            # offset = left_flowmask*x[:, :-1, :fold]
            # map_ = torch.norm(offset[0:1,0,:fold], dim=1)[0].cpu().detach().numpy()
            # plt.matshow(map_)
            # plt.axis('off')
            # plt.savefig('offset.png', bbox_inches='tight', pad_inches = 0)
            # plt.clf()
            # plt.close()

            # mask = left_flowmask[0,0,0,:,:].cpu().detach().numpy()
            # plt.matshow(mask)
            # plt.axis('off')
            # plt.savefig('mask.png', bbox_inches='tight', pad_inches = 0)
            # plt.clf()
            # plt.close()


            # map_ = torch.norm(x[0:1,0,0:fold], dim=1)[0].cpu().detach().numpy()
            # plt.matshow(map_)
            # plt.axis('off')
            # plt.savefig('center.png', bbox_inches='tight', pad_inches = 0)
            # plt.clf()
            # plt.close()
            # exit()        

        return out.transpose(1,2).contiguous()

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x, flows):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        B,C,L,H,W = x.shape
        x = x.transpose(1,2)

        fold = C // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.transpose(1,2)


class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-1] = input.data[:, 1:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
        input.data[:, :, fold: 2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None


class TemporalPool(nn.Module):
    def __init__(self, net, n_segment):
        super(TemporalPool, self).__init__()
        self.net = net
        self.n_segment = n_segment

    def forward(self, x):
        x = self.temporal_pool(x, n_segment=self.n_segment)
        return self.net(x)

    @staticmethod
    def temporal_pool(x, n_segment):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
        return x


def make_temporal_shift(net, n_segment, n_div=8, place='blockres', temporal_pool=False, gated=False, use_flow_tsm=False):
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))

    if gated == True and use_flow_tsm == True:
        ShiftModule = FlowGatedTemporalShift
    elif gated == True:
        ShiftModule = GatedTemporalShift
    else:
        ShiftModule = TemporalShift

    import model.tsm.resnet as resnet
    if isinstance(net, resnet.ResNet):
        if place == 'block':
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i] = ShiftModule(b, n_segment=this_segment, n_div=n_div)
                return Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

        elif 'blockres' in place:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
                print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1 = ShiftModule(b.conv1, n_segment=this_segment, n_div=n_div)
                return Sequential(*blocks)

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])
    else:
        raise NotImplementedError(place)


def make_temporal_pool(net, n_segment):
    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        print('=> Injecting nonlocal pooling')
        net.layer2 = TemporalPool(net.layer2, n_segment)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    # test inplace shift v.s. vanilla shift
    tsm1 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=False)
    tsm2 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=True)

    print('=> Testing CPU...')
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224)
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224)
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5

    print('=> Testing GPU...')
    tsm1.cuda()
    tsm2.cuda()
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224).cuda()
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224).cuda()
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5
    print('Test passed.')




