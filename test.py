#!/user/bin/python
# -*- encoding: utf-8 -*-

import os
import torch
import torchvision
from PIL import Image
from os.path import join, isdir
import numpy as np
from tqdm import tqdm
import cv2
from configs import Config  # 自己写了个用来存一些参数的

from utils import slide_inference


def test(cfg, model, test_loader, swin_save_dir):
    model.eval()
    dl = tqdm(test_loader)
    if not isdir(swin_save_dir):
        os.makedirs(swin_save_dir)
    for image, pth in dl:
        dl.set_description("Single-scale test")
        image = image.cuda()
        _, _, H, W = image.shape
        filename = pth[0]
        with torch.no_grad():
            swin_results = slide_inference(model, image)
            if cfg.side_edge:
                swin_results_all = torch.zeros((len(swin_results), 1, H, W))
                for i in range(len(swin_results)):
                    swin_results_all[i, 0, :, :] = swin_results[i]
                torchvision.utils.save_image((1 - swin_results_all), join(swin_save_dir, "%s.jpg" % filename), nrow=4)

            swin_result = torch.squeeze(swin_results[-1].detach()).cpu().numpy()
            swin_result = Image.fromarray((swin_result * 255).astype(np.uint8))
            swin_result.save(join(swin_save_dir, "%s.png" % filename))


def multiscale_test(model, test_loader, save_dir):
    model.eval()
    dl = tqdm(test_loader)
    if not isdir(save_dir):
        os.makedirs(save_dir)
    scale = [0.5, 1, 1.5]
    with torch.no_grad():
        for image, pth in dl:
            dl.set_description("Single-scale test")
            image = image[0]
            image_in = image.numpy().transpose((1, 2, 0))
            _, H, W = image.shape
            multi_fuse = np.zeros((H, W), np.float32)
            for k in range(0, len(scale)):
                im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
                im_ = im_.transpose((2, 0, 1))
                results = slide_inference(model, torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
                result = torch.squeeze(results[-1].detach()).cpu().numpy()
                fuse = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
                multi_fuse += fuse
            multi_fuse = multi_fuse / len(scale)
            # rescale trick suggested by jiangjiang
            multi_fuse = (multi_fuse - multi_fuse.min()) / (multi_fuse.max() - multi_fuse.min())
            filename = pth[0]
            result_out = Image.fromarray(((1 - multi_fuse) * 255).astype(np.uint8))
            result_out.save(join(save_dir, "%s.jpg" % filename))
            result_out_test = Image.fromarray((multi_fuse * 255).astype(np.uint8))
            result_out_test.save(join(save_dir, "%s.png" % filename))
