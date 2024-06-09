## environment
torch        2.0.0+cu118
torchaudio   2.0.0+cu118
torchvision  0.15.1+cu118

If you are prompted for no packages **, please enter pip install * * to install dependent package

### Data preparation
- Download the  [BSDS500](http://vcl.ucsd.edu/hed/HED-BSDS.tar) ,the [NYUDv2](http://vcl.ucsd.edu/hed/nyu/) provided by [HED](https://github.com/s9xie/hed) 
- and the [Multicue](https://serre-lab.clps.brown.edu/resource/multicue/)

### Pretrained Models
- Download the pretrained model on quark disk
- [Pretrained model for BSDS500](https://pan.quark.cn/s/2859b320f844)
- [Pretrained model for NYUDv2](https://pan.quark.cn/s/c60610c63b6b)

### Training
- Download the pre-trained [swin-transformer](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth) model, and put it to "./pretrained" folder.

## Acknowledgment
We acknowledge the effort from the authors of HED, CATS, EDTER on edge detection. Their researches laid the foundation for this work.
Our code reference [CATS](https://github.com/WHUHLX/CATS) and [https://github.com/microsoft/Swin-Transformer?tab=readme-ov-file](https://github.com/microsoft/Swin-Transformer?tab=readme-ov-file). Many thanks for their great work.  

```
@article{xie2017hed,
author = {Xie, Saining and Tu, Zhuowen},
journal = {International Journal of Computer Vision},
number = {1},
pages = {3--18},
title = {Holistically-Nested Edge Detection},
volume = {125},
year = {2017}
}

@article{huan2021unmixing,
	title={Unmixing convolutional features for crisp edge detection},
	author={Huan, Linxi and Xue, Nan and Zheng, Xianwei and He, Wei and Gong, Jianya and Xia, Gui-Song},
	journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
	volume={44},
	number={10},
	pages={6602--6609},
	year={2021},
	publisher={IEEE}
}

@inproceedings{pu2022edter,
	title={Edter: Edge detection with transformer},
	author={Pu, Mengyang and Huang, Yaping and Liu, Yuming and Guan, Qingji and Ling, Haibin},
	booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
	pages={1402--1412},
	year={2022}
}

@inproceedings{liu2021swin,
	title={Swin transformer: Hierarchical vision transformer using shifted windows},
	author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
	booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
	pages={10012--10022},
	year={2021}
}
```

