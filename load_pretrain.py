import torch

from configs import Config
from os.path import isfile

cfg = Config()


def load_pretrained(model):
    assert isfile(cfg.pretrain), "No checkpoint is found at '{}'".format(cfg.pretrain)

    model_weights_path_swin = cfg.pretrain

    pretrained_weights = torch.load(model_weights_path_swin)['state_dict']

    model.load_state_dict(pretrained_weights, strict=False)

    return model

