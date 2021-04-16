  
import matplotlib.patches as patches
from matplotlib.path import Path
import os
import sys
import io
import cv2
import time
import argparse
import shutil
import random
import zipfile
from glob import glob
import math
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageDraw, ImageFilter

import torch
import torchvision
import torch.nn as nn
import torch.distributed as dist

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('agg')


def create_random_shape_with_random_motion(video_length, imageHeight=240, imageWidth=432):
    # get a random shape
    height = random.randint(imageHeight//3, imageHeight-1)
    width = random.randint(imageWidth//3, imageWidth-1)
    edge_num = random.randint(6, 8)
    ratio = random.randint(6, 8)/10
    region = get_random_shape(
        edge_num=edge_num, ratio=ratio, height=height, width=width)
    region_width, region_height = region.size
    # get random position
    x, y = random.randint(
        0, imageHeight-region_height), random.randint(0, imageWidth-region_width)
    velocity = get_random_velocity(max_speed=3)
    m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
    m.paste(region, (y, x, y+region.size[0], x+region.size[1]))
    m = Image.fromarray(255 - np.array(m))
    masks = [m.convert('L')]

    # return fixed masks
    # if random.uniform(0, 1) > 0.5:
    return masks*video_length

    # return moving masks
    # for _ in range(video_length-1):
    #     x, y, velocity = random_move_control_points(
    #         x, y, imageHeight, imageWidth, velocity, region.size, maxLineAcceleration=(3, 0.5), maxInitSpeed=3)
    #     m = Image.fromarray(
    #         np.zeros((imageHeight, imageWidth)).astype(np.uint8))
    #     m.paste(region, (y, x, y+region.size[0], x+region.size[1]))
    #     m = Image.fromarray(255 - np.array(m))
    #     masks.append(m.convert('L'))
    # return masks


def get_random_shape(edge_num=9, ratio=0.7, width=432, height=240):
    '''
      There is the initial point and 3 points per cubic bezier curve. 
      Thus, the curve will only pass though n points, which will be the sharp edges.
      The other 2 modify the shape of the bezier curve.
      edge_num, Number of possibly sharp edges
      points_num, number of points in the Path
      ratio, (0, 1) magnitude of the perturbation from the unit circle, 
    '''
    points_num = edge_num*3 + 1
    angles = np.linspace(0, 2*np.pi, points_num)
    codes = np.full(points_num, Path.CURVE4)
    codes[0] = Path.MOVETO
    # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
    verts = np.stack((np.cos(angles), np.sin(angles))).T * \
        (2*ratio*np.random.random(points_num)+1-ratio)[:, None]
    verts[-1, :] = verts[0, :]
    path = Path(verts, codes)
    # draw paths into images
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='black', lw=2)
    ax.add_patch(patch)
    ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.axis('off')  # removes the axis to leave only the shape
    fig.canvas.draw()
    # convert plt images into numpy images
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape((fig.canvas.get_width_height()[::-1] + (3,)))
    plt.close(fig)
    # postprocess
    data = cv2.resize(data, (width, height))[:, :, 0]
    data = (1 - np.array(data > 0).astype(np.uint8))*255
    corrdinates = np.where(data > 0)
    xmin, xmax, ymin, ymax = np.min(corrdinates[0]), np.max(
        corrdinates[0]), np.min(corrdinates[1]), np.max(corrdinates[1])
    region = Image.fromarray(data).crop((ymin, xmin, ymax, xmax))
    return region


def random_accelerate(velocity, maxAcceleration, dist='uniform'):
    speed, angle = velocity
    d_speed, d_angle = maxAcceleration
    if dist == 'uniform':
        speed += np.random.uniform(-d_speed, d_speed)
        angle += np.random.uniform(-d_angle, d_angle)
    elif dist == 'guassian':
        speed += np.random.normal(0, d_speed / 2)
        angle += np.random.normal(0, d_angle / 2)
    else:
        raise NotImplementedError(
            f'Distribution type {dist} is not supported.')
    return (speed, angle)


def get_random_velocity(max_speed=3, dist='uniform'):
    if dist == 'uniform':
        speed = np.random.uniform(max_speed)
    elif dist == 'guassian':
        speed = np.abs(np.random.normal(0, max_speed / 2))
    else:
        raise NotImplementedError(
            f'Distribution type {dist} is not supported.')
    angle = np.random.uniform(0, 2 * np.pi)
    return (speed, angle)


def random_move_control_points(X, Y, imageHeight, imageWidth, lineVelocity, region_size, maxLineAcceleration=(3, 0.5), maxInitSpeed=3):
    region_width, region_height = region_size
    speed, angle = lineVelocity
    X += int(speed * np.cos(angle))
    Y += int(speed * np.sin(angle))
    lineVelocity = random_accelerate(
        lineVelocity, maxLineAcceleration, dist='guassian')
    if ((X > imageHeight - region_height) or (X < 0) or (Y > imageWidth - region_width) or (Y < 0)):
        lineVelocity = get_random_velocity(maxInitSpeed, dist='guassian')
    new_X = np.clip(X, 0, imageHeight - region_height)
    new_Y = np.clip(Y, 0, imageWidth - region_width)
    return new_X, new_Y, lineVelocity

import glob
import os

video_root = "/data2/xueyan-data/DAVIS/Train/JPEGImages"
out_root = "/data2/xueyan-data/DAVIS/Train/stationary_mask"

video_folders = sorted(glob.glob(os.path.join(video_root, '*')))
for video_folder in video_folders:
    image_pths = sorted(glob.glob(os.path.join(video_folder, "*.jpg")))
    folder_name = video_folder.split('/')[-1]
    image_names = [x.split('/')[-1] for x in image_pths]
    out_folder = os.path.join(out_root, folder_name)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    masks = create_random_shape_with_random_motion(len(image_names))
    for index, mask in enumerate(masks):
        x = np.array(mask)
        mask = mask.convert('1')
        mask.save(os.path.join(out_folder, image_names[index][:-4] + '.png'))

    print(video_folder)