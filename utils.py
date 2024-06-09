#!/user/bin/python
# -*- encoding: utf-8 -*-

import os, sys
import torch
import torch.nn as nn
import numpy as np
from os.path import join
import torch.nn.functional as F

from configs import Config

cfg = Config()


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def load_pretrained(model, fname, optimizer=None):
    """
    resume training from previous checkpoint
    :param fname: filename(with path) of checkpoint file
    :return: model, optimizer, checkpoint epoch
    """
    if os.path.isfile(fname):
        print("=> loading checkpoint '{}'".format(fname))
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            return model, optimizer, checkpoint['epoch']
        else:
            return model, checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(fname))


def slide_inference(model, img, stride=(180, 180), crop_size=(224, 224), num_classes=1, mask=None):
    """Inference by sliding-window with overlap.

    If h_crop > h_img or w_crop > w_img, the small patch will be used to
    decode without padding.
    """

    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    batch_size, _, h_img, w_img = img.size()
    # num_classes = self.num_classes
    if h_img < 224 or w_img < 224:
        mask = img.new_zeros((batch_size, num_classes, h_img, w_img))
        if h_img < 224:
            img = F.pad(img, (0, 0, 0, 224 - h_img), 'constant', 0)
        if w_img < 224:
            img = F.pad(img, (0, 224 - w_img, 0, 0), 'constant', 0)

        batch_size, _, h_img, w_img = img.size()

    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
    preds_swin_list = [preds.clone() for _ in range(cfg.decoder_channel + 1)]
    count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]
            swin_outputs = model(crop_img)
            for i in range(swin_outputs.__len__()):
                preds_swin_list[i] += F.pad(swin_outputs[i],
                                            (int(x1), int(preds.shape[3] - x2), int(y1),
                                             int(preds.shape[2] - y2)))

            count_mat[:, :, y1:y2, x1:x2] += 1
    assert (count_mat == 0).sum() == 0
    if torch.onnx.is_in_onnx_export():
        # cast count_mat to constant while exporting to ONNX
        count_mat = torch.from_numpy(
            count_mat.cpu().detach().numpy()).to(device=img.device)

    for i in range(preds_swin_list.__len__()):
        preds_swin_list[i] = preds_swin_list[i] / count_mat
        if mask is not None:
            preds_swin_list[i] = (preds_swin_list[i])[:, :, :mask.shape[2], :mask.shape[3]]

    return preds_swin_list
