## environment
torch        2.0.0+cu118
torchaudio   2.0.0+cu118
torchvision  0.15.1+cu118

If you are prompted for no packages **, please enter pip install * * to install dependent package

### Data preparation
- Download the  [BSDS500](http://vcl.ucsd.edu/hed/HED-BSDS.tar) ,the [NYUDv2](http://vcl.ucsd.edu/hed/nyu/) provided by [HED](https://github.com/s9xie/hed) 
- and the [Multicue](https://serre-lab.clps.brown.edu/resource/multicue/)

## Train
Enter python train.py to run the code. If you are prompted for no packages, enter pip install * * to install dependent packages

## Inference
After training, you can use the python test.py to validate your model.

## Citation
```
@article{li2024MSSwin,
  title={MS-Swin: Multiple Scale bi-direction fusion Swin Transformer for edge detection},
  author={Zhan Li, Zichang Wang, Han Zhang, Yuning Wang, Jin Xue, Xiaolong Ma, Yongqin Zhang},
  journal={},
  pages={},
  year={2024},
  publisher={IEEE}
}
```

