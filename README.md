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
@article{li2025SwinEdge,
  title={SwinEdge: Bidirectional Multscale Swin Feature Fusion for Edge Detection},
  author={Zhan Li, Han Zhanga,Zichang Wang, Yuning Wang, Hui Fang, Xiaolong Ma, Xunli Fan},
  journal={},
  pages={},
  year={2025},
  publisher={Elesiver}
}
```

