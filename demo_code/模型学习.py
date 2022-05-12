import os
import numpy as np
import rawpy
import torch
import skimage.metrics
from matplotlib import pyplot as plt
from unetTorch import Unet
import argparse


import glob
output_dir = '../data/out1'
imgnames = glob.glob("../data/testset/*.dng")
for img_name in imgnames:
    print(img_name)
    input_path = img_name
    tag = os.path.basename(img_name)[-5] #5
    output_path = os.path.join(output_dir, 'denoise'+ tag +'.dng')







img, height, width = read_image('../data/noise/0_noise.dng')
imgnp = img[0]
img2 = rawpy.imread('../data/noise/0_noise.dng')
img2data = img2.raw_image_visible


black_level = 1024
white_level = 16383
image2, height, width = read_image(os.path.join('../data/noise/0_noise.dng'))
image, height, width = read_image(os.path.join('../data/noise/0_noise.dng'))

# 标准化
image = normalization(image, black_level, white_level)
image = torch.from_numpy(image).float()
print(image.shape)
# batchsiz， 通道数， 高， 宽
image = image.view(-1, height//2, width//2, 4).permute(0, 3, 1, 2)
print(image.shape)
image = image.cpu().detach().numpy().transpose(0, 2, 3, 1)
print(image.shape)
# 反标准化
image = inv_normalization(image, black_level, white_level)
# 输出image信息
imgw = write_image(image, height, width)
print(imgw)
print(img2data)