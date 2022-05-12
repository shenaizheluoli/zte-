import os
import numpy as np
import rawpy
import torch
import skimage.metrics
from matplotlib import pyplot as plt
from unetTorch import Unet
import argparse


"""
值域[1024,16383]
"""
def normalization(input_data, black_level, white_level):
    output_data = (input_data.astype(float) - black_level) / (white_level - black_level)
    return output_data


def inv_normalization(input_data, black_level, white_level):
    output_data = np.clip(input_data, 0., 1.) * (white_level - black_level) + black_level
    output_data = output_data.astype(np.uint16)
    return output_data


def write_back_dng(src_path, dest_path, raw_data):
    """
    replace dng data
    """
    width = raw_data.shape[0]
    height = raw_data.shape[1]
    falsie = os.path.getsize(src_path)
    data_len = width * height * 2
    header_len = 8

    with open(src_path, "rb") as f_in:
        data_all = f_in.read(falsie)
        dng_format = data_all[5] + data_all[6] + data_all[7]

    with open(src_path, "rb") as f_in:
        header = f_in.read(header_len)
        if dng_format != 0:
            _ = f_in.read(data_len)
            meta = f_in.read(falsie - header_len - data_len)
        else:
            meta = f_in.read(falsie - header_len - data_len)
            _ = f_in.read(data_len)

        data = raw_data.tobytes()

    with open(dest_path, "wb") as f_out:
        f_out.write(header)
        if dng_format != 0:
            f_out.write(data)
            f_out.write(meta)
        else:
            f_out.write(meta)
            f_out.write(data)

    if os.path.getsize(src_path) != os.path.getsize(dest_path):
        print("replace raw data failed, file size mismatch!")
    else:
        print("replace raw data finished")


def read_image(input_path):
    raw = rawpy.imread(input_path)
    raw_data = raw.raw_image_visible
    height = raw_data.shape[0]
    width = raw_data.shape[1]

    raw_data_expand = np.expand_dims(raw_data, axis=2)
    raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                        raw_data_expand[0:height:2, 1:width:2, :],
                                        raw_data_expand[1:height:2, 0:width:2, :],
                                        raw_data_expand[1:height:2, 1:width:2, :]), axis=2)
    return raw_data_expand_c, height, width


def write_image(input_data, height, width):
    output_data = np.zeros((height, width), dtype=np.uint16)
    for channel_y in range(2):
        for channel_x in range(2):
            output_data[channel_y:height:2, channel_x:width:2] = input_data[0:, :, :, 2 * channel_y + channel_x]
    return output_data


"""
data_augement
"""

import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch
import torch.distributions as tdist
import numpy as np


class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, image, label):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        n, c, h0, w0 = image.shape
        if self.pad_if_needed and h0 < self.size[1]:
            image = F.pad(image, (self.size[1] - h0, 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - h0, 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and w0 < self.size[0]:
            image = F.pad(image, (0, self.size[0] - w0), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - w0), self.fill, self.padding_mode)

        # i, j, h, w = self.get_params(image, self.size)
        # return F.crop(image, i, j, h, w), center_crop(label, i, j, h0, w0)
        return F.center_crop(image, [h0, w0]), F.center_crop(label, [h0, w0])


class PairCompose(transforms.Compose):
    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label


class PairToTensor(transforms.ToTensor):
    def __call__(self, pic, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(label)


def random_noise_levels():
    """Generates random noise levels from a log-log linear distribution."""
    log_min_shot_noise = np.log(0.0001)
    log_max_shot_noise = np.log(0.012)
    log_shot_noise = torch.FloatTensor(1).uniform_(log_min_shot_noise, log_max_shot_noise)
    shot_noise = torch.exp(log_shot_noise)

    line = lambda x: 2.18 * x + 1.20
    n = tdist.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([0.26]))
    log_read_noise = line(log_shot_noise) + n.sample()
    read_noise = torch.exp(log_read_noise)
    return shot_noise, read_noise


def add_noise(image, shot_noise=0.01, read_noise=0.0005):
    """Adds random shot (proportional to image) and read (independent) noise."""

    # image    = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
    variance = image * shot_noise + read_noise
    n = tdist.Normal(loc=torch.zeros_like(variance), scale=torch.sqrt(variance))
    noise = n.sample()
    out = image + noise
    # out      = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
    return out




""""
data_load
"""

from torch.utils.data import Dataset, DataLoader
class DngDataset(Dataset):

    def __init__(self, image_dir, transform=None, is_eval=False):
        self.image_dir = image_dir
        self.input_list = os.listdir(os.path.join(image_dir, 'noisy'))
        self.ground_list = os.listdir(os.path.join(image_dir, 'ground_truth'))

        self._check_image(self.input_list)
        self.input_list.sort()
        self._check_image(self.ground_list)
        self.ground_list.sort()
        self.transform = transform
        self.is_test = is_eval

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):

        black_level = 1024
        white_level = 16383
        image, height, width = read_image(os.path.join(self.image_dir, 'noisy', self.input_list[idx]))

        image = normalization(image, black_level, white_level)
        image = torch.from_numpy(image).float()
        ##add noise
        noise = random.random() < 0.5
        if noise:
            # shot_noise, read_noise = random_noise_levels()
            # image  = add_noise(image, shot_noise, read_noise)
            image = add_noise(image)
        image = image.view(-1, height // 2, width // 2, 4).permute(0, 3, 1, 2)

        # image = torch.from_numpy(np.transpose(image.reshape(-1, height//4, width//4, 16), (0, 3, 1, 2))).float()
        label, height, width = read_image(os.path.join(self.image_dir, 'ground_truth', self.ground_list[idx]))
        label = normalization(label, black_level, white_level)
        label = torch.from_numpy(np.transpose(label.reshape(-1, height // 2, width // 2, 4), (0, 3, 1, 2))).float()

        if self.transform:
            image, label = self.transform(image, label)

        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] == 'DS_Store':
                continue
            if splits[-1] not in ['dng']:
                raise ValueError


class DngTestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = os.path.join(image_dir, 'test')
        # self.image_list = os.listdir(os.path.join(image_dir, 'blur/'))
        self.input_list = os.listdir(self.image_dir)
        self._check_image(self.input_list)
        self.input_list.sort()
        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        black_level = 1024
        white_level = 16383
        image, height, width = read_image(os.path.join(self.image_dir, self.input_list[idx]))
        image = normalization(image, black_level, white_level)
        image = torch.from_numpy(np.transpose(image.reshape(-1, height // 2, width // 2, 4), (0, 3, 1, 2))).float()
        name = self.input_list[idx]
        return image, name

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] == 'DS_Store':
                continue
            if splits[-1] not in ['dng']:
                raise ValueError


def train_dataloader(path, batch_size=4, num_workers=0, use_transform=True):
    transform = None
    path = args.data_dir
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop((1736, 2312), pad_if_needed=True),
                PairRandomHorizontalFilp(p=0.5),
            ]
        )
    dataloader = DataLoader(
        DngDataset(os.path.join(path, 'train'), transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DngTestDataset(path),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DngDataset(os.path.join(path, 'valid'), is_eval=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return dataloader


parser = argparse.ArgumentParser()

# Directories
parser.add_argument('--model_name', default='UNet', choices=['UNet','MIMO-UNet', 'MIMO-UNetPlus'], type=str)
parser.add_argument('--data_dir', type=str, default='../data/')
parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)

# Train
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=2e-4) # default=1e-4
parser.add_argument('--weight_decay', type=float, default=1e-8) ## default=0, 5e-2
parser.add_argument('--num_epoch', type=int, default=400)
parser.add_argument('--print_freq', type=int, default=10)
parser.add_argument('--num_worker', type=int, default=4)
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--valid_freq', type=int, default=10)
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--lr_steps', type=list, default=[(x+2) * 40 for x in range(400//50)])

# Test
parser.add_argument('--test_model', type=str, default='../checkpoints/UNet/Best.pth')
parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

args = parser.parse_args(args=[])
args.model_save_dir = os.path.join('../checkpoints/', args.model_name)
args.result_dir = os.path.join('../results/', args.model_name, 'result_image/')
print(args)




# 深度学习过程中，需要制作训练集和验证集、测试集。

import os, random, shutil

def copyfile(filename, target_dir):
    """将⽂件复制到⽬标⽬录"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

def moveFile(gt_fileDir, noisy_fileDir, vaild_gt_tarDir, vaild_noisy_tarDir, train_gt_tarDir, train_noisy_tarDir):
    pathDir = os.listdir(gt_fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate = 0.1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print(sample)
    for name in pathDir:
        if name in sample:
            copyfile(gt_fileDir + name, vaild_gt_tarDir)
            name2 = name.replace('gt', 'noise')
            copyfile(noisy_fileDir + name2, vaild_noisy_tarDir)
        else:
            copyfile(gt_fileDir + name, train_gt_tarDir)
            name2 = name.replace('gt', 'noise')
            copyfile(noisy_fileDir + name2, train_noisy_tarDir)
    return

if __name__ == '__main__':
    gt_fileDir = "../data/ground_truth/"  # 源图片文件夹路径
    noisy_fileDir = "../data/noisy/"
    vaild_gt_tarDir = '../data/vaild/ground_truth/'  # 移动到新的文件夹路径
    vaild_noisy_tarDir = '../data/vaild/noisy/'
    train_gt_tarDir = '../data/train/ground_truth/'  # 移动到新的文件夹路径
    train_noisy_tarDir = '../data/train/noisy/'
    moveFile(gt_fileDir, noisy_fileDir, vaild_gt_tarDir, vaild_noisy_tarDir, train_gt_tarDir, train_noisy_tarDir)

































