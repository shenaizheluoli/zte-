import os
import numpy as np
import rawpy
import torch
import skimage.metrics
from matplotlib import pyplot as plt
from unetTorch import Unet
import argparse
import pylab

gt_path = '../data/ground_truth/1_gt.dng'
noise_path = '../data/noisy/1_noise.dng'
gt = rawpy.imread(gt_path).raw_image_visible
noise = rawpy.imread(noise_path).raw_image_visible

plt.imshow(gt)
plt.show()
plt.imshow(noise)
plt.show()

black_level = 1024
white_level = 16383
psnr = skimage.metrics.peak_signal_noise_ratio(
    gt.astype(np.float),
    noise.astype(np.float),
    data_range=white_level)
ssim = skimage.metrics.structural_similarity(
    gt.astype(np.float),
    noise.astype(np.float),
    multichannel=True,
    data_range=white_level)
print('psnr:', psnr)
print('ssim:', ssim)

ground_path = '../data/ground_truth/1_gt.dng'
input_path = '../data/noisy/1_noise.dng'

f0 = rawpy.imread(ground_path)
f1 = rawpy.imread(input_path)

f, axarr = plt.subplots(1, 2)
axarr[0].imshow(f0.postprocess(use_camera_wb=True))
axarr[1].imshow(f1.postprocess(use_camera_wb=True))

axarr[0].set_title('gt')
axarr[1].set_title('noisy')

plt.show()
