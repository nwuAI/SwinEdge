import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from model.upsample_block.upsample_block_up2down import Upsample_Block_up2down
from model.upsample_block.upsample_block_down2up import Upsample_Block_down2up


class Upsample_Block(nn.Module):
    def __init__(self, out_channel=[96, 192, 384, 768], mid_channels=128):
        super(Upsample_Block, self).__init__()
        self.out_channel = out_channel
        self.mid_channels = mid_channels

        self.ub_up2down = Upsample_Block_up2down(self.out_channel, self.mid_channels)
        self.ub_down2up = Upsample_Block_down2up(self.out_channel, self.mid_channels)

    def forward(self, x):
        ub_up2down_output = self.ub_up2down(x)
        ub_down2up_output = self.ub_down2up(x)

        ub_output = [ub_up2down_output[0], ub_up2down_output[1], ub_up2down_output[2], ub_up2down_output[3],
                     ub_down2up_output[0], ub_down2up_output[1], ub_down2up_output[2], ub_down2up_output[3]]

        return ub_output

