#!/user/bin/python
# -*- encoding: utf-8 -*-

from os.path import join


class Config(object):
    def __init__(self):
        self.data = "bsds"
        # ============== training
        self.pretrain = "./pretrained/your pretrain weight"
        self.msg_iter = 500
        self.gpu = '0'
        self.aug = True
        self.mode = 'train'

        # ============== testing
        self.multi_aug = True  # Produce the multi-scale results
        self.side_edge = True  # Output the side edges

        # ================ dataset
        self.dataset = "./data/{}".format(self.data)

        # =============== optimizer
        self.batch_size = 1
        self.lr = 1e-6
        self.momentum = 0.9
        self.wd = 2e-4
        self.stepsize = 5
        self.gamma = 0.1
        self.max_epoch = 30
        self.itersize = 10
        self.decoder_channel = 8

        self.num_heads = [6, 12, 24, 48]
        self.out_channel = [192, 384, 768, 1536]
        self.depths = [2, 2, 18, 2]
        self.embed_dim = 192
