#!/user/bin/python
# -*- encoding: utf-8 -*-

import os, sys
import numpy as np
from PIL import Image
import cv2
import shutil
import argparse
import time
import datetime
import torch
from torch import optim
from torch.optim import lr_scheduler

from data.data_loader import MyDataLoader
from load_pretrain import load_pretrained
from torch.utils.data import DataLoader, sampler

from model.model import SwinTransformer
from train import train
from utils import Logger, Averagvalue, save_checkpoint
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
from configs import Config
from test import test, multiscale_test

cfg = Config()

if cfg.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

THIS_DIR = abspath(dirname(__file__))
TMP_DIR = join('result save location')

if not isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


def main():
    # model
    model = SwinTransformer(embed_dim=cfg.embed_dim, depths=cfg.depths, num_heads=cfg.num_heads)
    model = load_pretrained(model)
    print('=> Load model')

    model.cuda()

    print('=> Cuda used')

    test_dataset = MyDataLoader(root=cfg.dataset, split="test")

    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False)

    if cfg.mode == "test":
        test(cfg, model, test_loader, swin_save_dir=join(TMP_DIR, "test", "sing_scale_test"))
        if cfg.multi_aug:
            multiscale_test(model, test_loader, save_dir=join(TMP_DIR, "test", "multi_scale_test"))

    else:
        train_dataset = MyDataLoader(root=cfg.dataset, split="train", transform=cfg.aug)

        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                  num_workers=1, drop_last=True, shuffle=True)

        model.train()

        # optimizer
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.wd)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.stepsize, gamma=cfg.gamma)

        # log
        log = Logger(join(TMP_DIR, "%s-%d-log.txt" % ("sgd", cfg.lr)))
        sys.stdout = log

        train_loss = []
        train_loss_detail = []

        for epoch in range(0, cfg.max_epoch):
            tr_avg_loss, tr_detail_loss = train(cfg, train_loader, model, optimizer, scheduler, epoch,
                                                swin_save_dir=join(TMP_DIR, "train",
                                                                   "epoch-%d-training-record" % epoch))

            test(model, test_loader,
                 swin_save_dir=join(TMP_DIR, "test", "epoch-%d-testing-record-view" % epoch))

            log.flush()

            train_loss.append(tr_avg_loss)
            train_loss_detail += tr_detail_loss


if __name__ == '__main__':
    main()
