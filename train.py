#!/user/bin/python
# -*- encoding: utf-8 -*-

import torch
import time

from tqdm import tqdm

from loss.crossentropyloss import crossentropyloss
from utils import Averagvalue, save_checkpoint, slide_inference
import os
from os.path import join, isdir
import torchvision


def train(cfg, train_loader, model, optimizer, scheduler, epoch, swin_save_dir):
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    # switch to train mode
    model.train()
    end = time.time()
    epoch_loss = []
    counter = 0
    for i, (image, label, pth) in enumerate(tqdm(train_loader)):

        # measure data loading time
        data_time.update(time.time() - end)
        image, label = image.cuda(), label.cuda()

        swin_outputs = slide_inference(model, image)
        loss = torch.zeros(1).cuda()

        for o in swin_outputs[:cfg.decoder_channel]:
            loss = loss + crossentropyloss(o, label)
        loss = loss + crossentropyloss(swin_outputs[cfg.decoder_channel], label) * 0.8

        counter += 1
        loss = loss / cfg.itersize
        loss.backward()

        if counter == cfg.itersize:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0

        # measure accuracy and record loss
        losses.update(loss.item(), image.size(0))
        epoch_loss.append(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        if not isdir(swin_save_dir):
            os.makedirs(swin_save_dir)

        if i % cfg.msg_iter == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, 30, i, len(train_loader)) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(
                       loss=losses)
            print(info)

            label_out = torch.eq(label, 1).float()

            swin_outputs.append(label_out)
            _, _, H, W = swin_outputs[0].shape
            swin_all_results = torch.zeros((len(swin_outputs), 1, H, W))

            for j in range(len(swin_outputs)):
                swin_all_results[j, 0, :, :] = swin_outputs[j][0, 0, :, :]

            torchvision.utils.save_image(1 - swin_all_results, join(swin_save_dir, "iter-%d.jpg" % i), nrow=4)

    # adjust lr
    scheduler.step()
    # save checkpoint
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, filename=join(swin_save_dir, "epoch-%d-checkpoint.pth" % epoch))

    return losses.avg, epoch_loss
