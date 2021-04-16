import numpy as np
import os
import cv2
import sys
import argparse
import math
import warnings
import torch
import torch.nn.functional as F

TAG_FLOAT = 202021.25

def read(file):
	assert type(file) is str, "file is not str %r" % str(file)
	assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
	assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
	f = open(file,'rb')
	flo_number = np.fromfile(f, np.float32, count=1)[0]
	assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
	w = np.fromfile(f, np.int32, count=1)
	h = np.fromfile(f, np.int32, count=1)
	#if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
	# data = np.fromfile(f, np.float32, count=2*w*h)
	data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
	# Reshape data into 3D array (columns, rows, bands)
	flow = np.resize(data, (int(h), int(w), 2))	
	f.close()
	return flow


def makeColorwheel():
	#  color encoding scheme

	#   adapted from the color circle idea described at
	#   http://members.shaw.ca/quadibloc/other/colint.htm

	RY = 15
	YG = 6
	GC = 4
	CB = 11
	BM = 13
	MR = 6

	ncols = RY + YG + GC + CB + BM + MR

	colorwheel = np.zeros([ncols, 3]) # r g b

	col = 0
	#RY
	colorwheel[0:RY, 0] = 255
	colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
	col += RY

	#YG
	colorwheel[col:YG+col, 0]= 255 - np.floor(255*np.arange(0, YG, 1)/YG)
	colorwheel[col:YG+col, 1] = 255;
	col += YG;

	#GC
	colorwheel[col:GC+col, 1]= 255 
	colorwheel[col:GC+col, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
	col += GC;

	#CB
	colorwheel[col:CB+col, 1]= 255 - np.floor(255*np.arange(0, CB, 1)/CB)
	colorwheel[col:CB+col, 2] = 255
	col += CB;

	#BM
	colorwheel[col:BM+col, 2]= 255 
	colorwheel[col:BM+col, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
	col += BM;

	#MR
	colorwheel[col:MR+col, 2]= 255 - np.floor(255*np.arange(0, MR, 1)/MR)
	colorwheel[col:MR+col, 0] = 255
	return 	colorwheel


def computeColor(u, v):
	colorwheel = makeColorwheel();
	nan_u = np.isnan(u)
	nan_v = np.isnan(v)
	nan_u = np.where(nan_u)
	nan_v = np.where(nan_v) 

	u[nan_u] = 0
	u[nan_v] = 0
	v[nan_u] = 0 
	v[nan_v] = 0

	ncols = colorwheel.shape[0]
	radius = np.sqrt(u**2 + v**2)
	a = np.arctan2(-v, -u) / np.pi
	fk = (a+1) /2 * (ncols-1) # -1~1 maped to 1~ncols
	k0 = fk.astype(np.uint8)	 # 1, 2, ..., ncols
	k1 = k0+1;
	k1[k1 == ncols] = 0
	f = fk - k0

	img = np.empty([k1.shape[0], k1.shape[1],3])
	ncolors = colorwheel.shape[1]
	for i in range(ncolors):
		tmp = colorwheel[:,i]
		col0 = tmp[k0]/255
		col1 = tmp[k1]/255
		col = (1-f)*col0 + f*col1
		idx = radius <= 1
		col[idx] = 1 - radius[idx]*(1-col[idx]) # increase saturation with radius    
		col[~idx] *= 0.75 # out of range
		img[:,:,2-i] = np.floor(255*col).astype(np.uint8)

	return img.astype(np.uint8)


def resize_flow(flow, size):
	# flow: [N, 2, H, W]
	# size: [h, w]
	h, w = size
	fh, fw = flow.size(-2), flow.size(-1)
	flow = F.interpolate(flow, [h, w], mode='nearest')
	flow[:, 0, :, :] = flow[:, 0, :, :] * w / fw
	flow[:, 1, :, :] = flow[:, 1, :, :] * h / fh
	return flow


def resize_flow_np(flow, size):
	# flow: h,w,2
	# size: h,w
	h,w = size
	fh, fw, _ = flow.shape
	flow = cv2.resize(flow, size[::-1], interpolation=cv2.INTER_NEAREST)
	flow[:, :, 0] = flow[:, :, 0] * w / fw
	flow[:, :, 1] = flow[:, :, 1] * h / fh
	return flow


def flow_to_grid(flow):
    # flow: [N, 2, H, W]
    N, _, H, W = flow.size()
    flow_x = flow[:, 0, ...]
    flow_y = flow[:, 1, ...]
    flow_x = flow_x / float(W) * 2.0
    flow_y = flow_y / float(H) * 2.0
    y, x = torch.meshgrid([torch.linspace(-1, 1, steps=H), torch.linspace(-1, 1, steps=W)])
    y, x = y[None, ...], x[None, ...]
    y = y.to(flow.device)
    x = x.to(flow.device)
    grid = torch.zeros(N, H, W, 2).to(flow.device)
    grid[..., 0] = x + flow_x
    grid[..., 1] = y + flow_y
    return grid.permute(0,3,1,2)


# save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project
def writeFlow(name, flow):
	f = open(name, 'wb')
	f.write('PIEH'.encode('utf-8'))
	np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
	flow = flow.astype(np.float32)
	flow.tofile(f)
	f.flush()
	f.close()


def writeFlowImage(save_dir, data):
    truerange = 1
    range_f = truerange * 1.04
    u, v = data[:, :, 0], data[:, :, 1]
    img = computeColor(u/range_f/math.sqrt(2), v/range_f/math.sqrt(2));
    cv2.imwrite(save_dir, img)


def compress_flow(flow, bound=96):
    # input size [H, W, 2]
    flow = flow.copy()
    H, W = flow.shape[0], flow.shape[1]
    
    min_flow, max_flow = np.min(flow), np.max(flow)
    if min_flow < -bound or max_flow > bound:
        warnings.warn('Min: %.4f, Max: %.4f, out of [-%d, %d]' % (min_flow, max_flow, bound, bound))
    
    flow[..., 0] = np.round((flow[..., 0] + bound) / (2. * bound) * 255.)
    flow[..., 1] = np.round((flow[..., 1] + bound) / (2. * bound) * 255.)
    flow[flow < 0] = 0
    flow[flow > 255] = 255
    flow = np.concatenate([flow, np.zeros([H, W, 1])], axis=2)
    flow = flow.astype(np.uint8)
    return flow


def decompress_flow(flow, bound=96):
    # input size [H, W, 2]
    flow = flow.copy().astype(np.float32)
    H, W = flow.shape[0], flow.shape[1]
    flow[..., 0] = flow[..., 0] / 255. * 2 * bound - bound
    flow[..., 1] = flow[..., 1] / 255. * 2 * bound - bound
    flow = flow[..., :2]
    return flow


def writeCompressFlow(name, flow):
	flow = compress_flow(flow)
	cv2.imwrite(name, flow)


def readCompressFlow(name):
	compress_flow = cv2.imread(name)
	# flow = decompress_flow(compress_flow)
	return compress_flow


def flip_flow(flow):
    # flow: [H, W, 2]
    flow = flow[:, ::-1, :]
    flow[..., 0] = - flow[..., 0]
    return flow