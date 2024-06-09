import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Upsample_Block_up2down_4(nn.Module):
    def __init__(self, out_channel=[96, 192, 384, 768]):
        super(Upsample_Block_up2down_4, self).__init__()
        self.out_channel_t = out_channel
        self.ub_up2down_4_1 = nn.Sequential(
            nn.ConvTranspose2d(out_channel[3], out_channel[2], 2, stride=2, bias=False),
            nn.BatchNorm2d(out_channel[2]), nn.ReLU())
        self.ub_up2down_4_2 = nn.Sequential(
            nn.ConvTranspose2d(out_channel[2], out_channel[1], 2, stride=2, bias=False),
            nn.BatchNorm2d(out_channel[1]), nn.ReLU())
        self.ub_up2down_4_3 = nn.Sequential(
            nn.ConvTranspose2d(out_channel[1], out_channel[0], 2, stride=2, bias=False),
            nn.BatchNorm2d(out_channel[0]), nn.ReLU())

    def forward(self, x):
        up2down4_output = []

        x = self.ub_up2down_4_1(x)
        up2down4_output.append(x)

        x = self.ub_up2down_4_2(x)
        up2down4_output.append(x)

        x = self.ub_up2down_4_3(x)
        up2down4_output.append(x)

        return up2down4_output


class Upsample_Block_up2down_3(nn.Module):
    def __init__(self, out_channel=[96, 192, 384, 768]):
        super(Upsample_Block_up2down_3, self).__init__()
        self.out_channel_t = out_channel
        self.ub_up2down_3_1 = nn.Sequential(
            nn.ConvTranspose2d(out_channel[2], out_channel[1], 2, stride=2, bias=False),
            nn.BatchNorm2d(out_channel[1]), nn.ReLU())
        self.ub_up2down_3_2 = nn.Sequential(
            nn.ConvTranspose2d(out_channel[1], out_channel[0], 2, stride=2, bias=False),
            nn.BatchNorm2d(out_channel[0]), nn.ReLU())

    def forward(self, x, x_output4):
        up2down3_output = []

        x = x + x_output4[0]

        x = self.ub_up2down_3_1(x)
        up2down3_output.append(x)

        x = self.ub_up2down_3_2(x)
        up2down3_output.append(x)

        return up2down3_output


class Upsample_Block_up2down_2(nn.Module):
    def __init__(self, out_channel=[96, 192, 384, 768]):
        super(Upsample_Block_up2down_2, self).__init__()
        self.out_channel_t = out_channel
        self.ub_up2down_2_1 = nn.Sequential(
            nn.ConvTranspose2d(out_channel[1], out_channel[0], 2, stride=2, bias=False),
            nn.BatchNorm2d(out_channel[0]), nn.ReLU())

    def forward(self, x, x_output4, x_output3):
        up2down2_output = []

        x = x + x_output3[0] + x_output4[1]

        x = self.ub_up2down_2_1(x)

        up2down2_output.append(x)

        return up2down2_output


class Upsample_Block_up2down(nn.Module):
    def __init__(self, out_channel=[96, 192, 384, 768], mid_channels=128):
        super(Upsample_Block_up2down, self).__init__()
        self.out_channel = out_channel

        self.ub_up2down_4 = Upsample_Block_up2down_4(self.out_channel)
        self.ub_up2down_3 = Upsample_Block_up2down_3(self.out_channel)
        self.ub_up2down_2 = Upsample_Block_up2down_2(self.out_channel)

        self.deconv1_down = nn.Sequential(
            nn.Conv2d(out_channel[0], out_channel[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel[0]), nn.ReLU(),
            nn.ConvTranspose2d(in_channels=out_channel[0], out_channels=mid_channels, kernel_size=8, stride=4,
                               padding=2,
                               bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(),
            nn.Conv2d(mid_channels, 1, 1))
        self.deconv2_down = nn.Sequential(
            nn.Conv2d(out_channel[0], out_channel[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel[0]), nn.ReLU(),
            nn.ConvTranspose2d(in_channels=out_channel[0], out_channels=mid_channels, kernel_size=8, stride=4,
                               padding=2,
                               bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(),
            nn.Conv2d(mid_channels, 1, 1))
        self.deconv3_down = nn.Sequential(
            nn.Conv2d(out_channel[0], out_channel[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel[0]), nn.ReLU(),
            nn.ConvTranspose2d(in_channels=out_channel[0], out_channels=mid_channels, kernel_size=8, stride=4,
                               padding=2,
                               bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(),
            nn.Conv2d(mid_channels, 1, 1))
        self.deconv4_down = nn.Sequential(
            nn.Conv2d(out_channel[0], out_channel[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel[0]), nn.ReLU(),
            nn.ConvTranspose2d(in_channels=out_channel[0], out_channels=mid_channels, kernel_size=8, stride=4,
                               padding=2,
                               bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(),
            nn.Conv2d(mid_channels, 1, 1))

    def forward(self, x):
        # x[0]时56x56的
        ub_up2down4_output = self.ub_up2down_4(x[3])
        ub_up2down3_output = self.ub_up2down_3(x[2], ub_up2down4_output)
        ub_up2down2_output = self.ub_up2down_2(x[1], ub_up2down4_output, ub_up2down3_output)
        ub_up2down1_output = x[0] + ub_up2down4_output[2] + ub_up2down3_output[1] + ub_up2down2_output[0]

        ub_up2down_output = [self.deconv1_down(ub_up2down1_output),
                          self.deconv2_down(ub_up2down2_output[0]),
                          self.deconv3_down(ub_up2down3_output[1]),
                          self.deconv4_down(ub_up2down4_output[2])]

        return ub_up2down_output
