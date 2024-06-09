import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Upsample_Block_down2up_1(nn.Module):
    def __init__(self, out_channel=[96, 192, 384, 768]):
        super(Upsample_Block_down2up_1, self).__init__()
        self.out_channel_t = out_channel
        self.ub_down2up1_1_sub = nn.Sequential(
            nn.Conv2d(out_channel[0], out_channel[1], 3, stride=2, padding=1),
            nn.ReLU())
        self.ub_down2up1_2_sub = nn.Sequential(
            nn.Conv2d(out_channel[1], out_channel[2], 3, stride=2, padding=1),
            nn.ReLU())
        self.ub_down2up1_3_sub = nn.Sequential(
            nn.Conv2d(out_channel[2], out_channel[3], 3, stride=2, padding=1),
            nn.ReLU())

    def forward(self, x):
        ub_down2up1_sub_output = []

        x = self.ub_down2up1_1_sub(x)
        ub_down2up1_sub_output.append(x)

        x = self.ub_down2up1_2_sub(x)
        ub_down2up1_sub_output.append(x)

        x = self.ub_down2up1_3_sub(x)
        ub_down2up1_sub_output.append(x)

        return ub_down2up1_sub_output


class Upsample_Block_down2up_2(nn.Module):
    def __init__(self, out_channel=[96, 192, 384, 768]):
        super(Upsample_Block_down2up_2, self).__init__()
        self.out_channel_t = out_channel

        self.ub_down2up2_1_sub = nn.Sequential(
            nn.Conv2d(out_channel[1], out_channel[2], 3, stride=2, padding=1),
            nn.ReLU())

        self.ub_down2up2_2_sub = nn.Sequential(
            nn.Conv2d(out_channel[2], out_channel[3], 3, stride=2, padding=1),
            nn.ReLU())

        self.ub_up2down_2_1 = nn.Sequential(
            nn.ConvTranspose2d(out_channel[1], out_channel[0], 2, stride=2, bias=False),
            nn.BatchNorm2d(out_channel[0]), nn.ReLU())

    def forward(self, x, ub_down2up1_sub_output):
        ub_down2up2_sub_output = []
        x_up = x

        x = self.ub_down2up2_1_sub(x)
        ub_down2up2_sub_output.append(x)

        x = self.ub_down2up2_2_sub(x)
        ub_down2up2_sub_output.append(x)

        ub_down2up2_output = []
        x_up = ub_down2up1_sub_output[0] + x_up
        x_up = self.ub_up2down_2_1(x_up)
        ub_down2up2_output.append(x_up)

        return ub_down2up2_sub_output, ub_down2up2_output


class Upsample_Block_down2up_3(nn.Module):
    def __init__(self, out_channel=[96, 192, 384, 768]):
        super(Upsample_Block_down2up_3, self).__init__()
        self.out_channel_t = out_channel

        self.ub_down2up3_1_sub = nn.Sequential(
            nn.Conv2d(out_channel[2], out_channel[3], 3, stride=2, padding=1),
            nn.ReLU())

        self.ub_up2down_3_1 = nn.Sequential(
            nn.ConvTranspose2d(out_channel[2], out_channel[1], 2, stride=2, bias=False),
            nn.BatchNorm2d(out_channel[1]), nn.ReLU())

        self.ub_up2down_3_2 = nn.Sequential(
            nn.ConvTranspose2d(out_channel[1], out_channel[0], 2, stride=2, bias=False),
            nn.BatchNorm2d(out_channel[0]), nn.ReLU())

    def forward(self, x, ub_down2up1_sub_output, ub_down2up2_sub_output):
        ub_down2up3_sub_output = []
        x_up = x

        x = self.ub_down2up3_1_sub(x)
        ub_down2up3_sub_output.append(x)

        ub_down2up3_output = []
        x_up = ub_down2up2_sub_output[0] + ub_down2up1_sub_output[1] + x_up
        x_up = self.ub_up2down_3_1(x_up)
        ub_down2up3_output.append(x_up)

        x_up = self.ub_up2down_3_2(x_up)
        ub_down2up3_output.append(x_up)

        return ub_down2up3_sub_output, ub_down2up3_output


class Upsample_Block_down2up_4(nn.Module):
    def __init__(self, out_channel=[96, 192, 384, 768]):
        super(Upsample_Block_down2up_4, self).__init__()
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

    def forward(self, x, ub_down2up1_sub_output, ub_down2up2_sub_output, ub_down2up3_sub_output):
        ub_down2up3_output = []
        x_up = x + ub_down2up3_sub_output[0] + ub_down2up2_sub_output[1] + ub_down2up1_sub_output[2]
        x_up = self.ub_up2down_4_1(x_up)
        ub_down2up3_output.append(x_up)

        x_up = self.ub_up2down_4_2(x_up)
        ub_down2up3_output.append(x_up)

        x_up = self.ub_up2down_4_3(x_up)
        ub_down2up3_output.append(x_up)

        return ub_down2up3_output


class Upsample_Block_down2up(nn.Module):
    def __init__(self, out_channel=[96, 192, 384, 768], mid_channels=128):
        super(Upsample_Block_down2up, self).__init__()
        self.out_channel = out_channel

        self.ub_down2up_1 = Upsample_Block_down2up_1(self.out_channel)
        self.ub_down2up_2 = Upsample_Block_down2up_2(self.out_channel)
        self.ub_down2up_3 = Upsample_Block_down2up_3(self.out_channel)
        self.ub_down2up_4 = Upsample_Block_down2up_4(self.out_channel)

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
        ub_down2up1_sub_output = self.ub_down2up_1(x[0])
        ub_down2up2_sub_output, ub_down2up2_output = self.ub_down2up_2(x[1], ub_down2up1_sub_output)
        ub_down2up3_sub_output, ub_down2up3_output = self.ub_down2up_3(x[2], ub_down2up1_sub_output,
                                                                       ub_down2up2_sub_output)
        ub_down2up4_output = self.ub_down2up_4(x[3], ub_down2up1_sub_output, ub_down2up2_sub_output,
                                               ub_down2up3_sub_output)

        ub_down2up_output = [self.deconv1_down(x[0]),
                             self.deconv2_down(ub_down2up2_output[0]),
                             self.deconv3_down(ub_down2up3_output[1]),
                             self.deconv4_down(ub_down2up4_output[2])]

        return ub_down2up_output
