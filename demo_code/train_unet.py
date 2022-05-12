import os
import torch

from dataloader.data_loader import train_dataloader,test_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Unet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(Unet, self).__init__()

        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv10_1 = nn.Conv2d(32, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        n, c, h, w = x.shape
        h_pad = 32 - h % 32 if not h % 32 == 0 else 0
        w_pad = 32 - w % 32 if not w % 32 == 0 else 0
        padded_image = F.pad(x, (0, w_pad, 0, h_pad), 'replicate')

        conv1 = self.leaky_relu(self.conv1_1(padded_image))
        conv1 = self.leaky_relu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)

        conv2 = self.leaky_relu(self.conv2_1(pool1))
        conv2 = self.leaky_relu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)

        conv3 = self.leaky_relu(self.conv3_1(pool2))
        conv3 = self.leaky_relu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)

        conv4 = self.leaky_relu(self.conv4_1(pool3))
        conv4 = self.leaky_relu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)

        conv5 = self.leaky_relu(self.conv5_1(pool4))
        conv5 = self.leaky_relu(self.conv5_2(conv5))

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.leaky_relu(self.conv6_1(up6))
        conv6 = self.leaky_relu(self.conv6_2(conv6))

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.leaky_relu(self.conv7_1(up7))
        conv7 = self.leaky_relu(self.conv7_2(conv7))

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.leaky_relu(self.conv8_1(up8))
        conv8 = self.leaky_relu(self.conv8_2(conv8))

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.leaky_relu(self.conv9_1(up9))
        conv9 = self.leaky_relu(self.conv9_2(conv9))

        conv10 = self.conv10_1(conv9)
        out = conv10[:, :, :h, :w]

        return out

    def leaky_relu(self, x):
        out = torch.max(0.2 * x, x)
        return out


def _train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mse_criterion = torch.nn.MSELoss(reduce=True, size_average=True, reduction='none')
    l1_criterion = torch.nn.L1Loss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay,
                                 betas=(0.9, 0.999))
    dataloader = train_dataloader(args.data_dir, args.batch_size, 0)
    max_iter = len(dataloader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)
    epoch = 1
    writer = SummaryWriter()
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr = -1
    best_ssim = -1
    for epoch_idx in range(epoch, args.num_epoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):
            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            input_img = input_img.squeeze(dim = 1)
            label_img = label_img.squeeze(dim = 1)

            optimizer.zero_grad()
            pred_img = model(input_img)

            loss_content = mse_criterion(pred_img, label_img)
            label_fft = torch.fft.fft2(label_img, dim=(-2, -1))
            pred_fft = torch.fft.fft2(pred_img, dim=(-2, -1))
            loss_fft = l1_criterion(pred_fft, label_fft)

            loss = 0.8 * loss_content + 0.2 * loss_fft
            loss.backward()
            optimizer.step()

            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())

            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())

            if (iter_idx + 1) % args.print_freq == 0:
                lr = check_lr(optimizer)
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, lr, iter_pixel_adder.average(),
                    iter_fft_adder.average()))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()

        if epoch_idx > 100 and  epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pth' % epoch_idx)
            torch.save({'model': model.state_dict()}, save_name)
            #if need resume, use this method to save model
            # torch.save({'model': model.state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #             'scheduler': scheduler.state_dict(),
            #             'epoch': epoch_idx}, save_name)
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average()))
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        scheduler.step()
        if epoch_idx % 1 == 0:
            psnr,ssim = _valid(model, args, epoch_idx)
            print('%03d epoch \n Average GOPRO PSNR %.2f dB SSIM %.2f' % (epoch_idx, psnr,ssim))
            writer.add_scalar('PSNR_GOPRO', psnr, epoch_idx)
            writer.add_scalar('SSIM_GOPRO', ssim, epoch_idx)
            if psnr >= best_psnr and ssim >= best_ssim:
                best_psnr = psnr
                best_ssim = ssim
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'best_model.pth'))
    save_name = os.path.join(args.model_save_dir, 'final.pth')
    torch.save({'model': model.state_dict()}, save_name)

model = Unet()


def train(DataLoader, model, criterion, optimizer):
    model.cuda()
    # 指定为train模式
    model.train()

    for i, (img, target) in enumerate(DataLoader):
        img = img.cuda()
        target = target.cuda()
        # 计算网络输出
        output = model(img)

        # 计算损失
        loss = criterion(output, target)

    # 计算梯度和做反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



