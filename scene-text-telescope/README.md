# Scene Text Telescope

This is the code for CVPR2021 paper "Scene Text Telescope: Text-Focused Scene Image Super-Resolution". [[link]](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Scene_Text_Telescope_Text-Focused_Scene_Image_Super-Resolution_CVPR_2021_paper.html)

![architecture](./image/architecture.png)


## Dependencies
Build up an environment with python3.6, and download corresponding libraries with pip
```python
pip install -r requirement.txt
```

## Pre-trained Model
Here are some outputs with backbone **TBSRN** while text-focused

[[Model]](https://drive.google.com/file/d/13SH_wy1QRQoWWHxI8lF0VqtZYTXBhhKy/view?usp=sharing) [[Log]](https://drive.google.com/file/d/19M5twD_cUAq88YuENPpR_7hIvLULb6mF/view?usp=sharing)


## Dataset
Download all resources at [BaiduYunDisk](https://pan.baidu.com/s/1P_SCcQG74fiQfTnfidpHEw) with password: stt6, or [Dropbox](https://www.dropbox.com/sh/f294n405ngbnujn/AABUO6rv_5H5MvIvCblcf-aKa?dl=0)

* TextZoom dataset
* Pretrained weights of CRNN 
* Pretrained weights of Transformer-based recognizer

All the resources shoulded be placed under ```./dataset/mydata```, for example
```python
./dataset/mydata/train1
./dataset/mydata/train2
./dataset/mydata/pretrain_transformer.pth
...
```


## Training
Please remember to modify the experiment name. Two text-focused modules are activated whenever ```--text_focus``` is used
```python
CUDA_VISIBLE_DEVICES=GPU_NUM python main.py --batch_size=16 --STN --exp_name EXP_NAME --text_focus
```

## Testing
```python
CUDA_VISIBLE_DEVICES=GPU_NUM python main.py --batch_size=16 --STN --exp_name EXP_NAME --text_focus --resume YOUR_MODEL --test --test_data_dir ./dataset/mydata/test
```

## Demo
```python
CUDA_VISIBLE_DEVICES=GPU_NUM python main.py --batch_size=16 --STN --exp_name EXP_NAME --text_focus --demo --demo_dir ./demo
```

## Acknowledgement
We inherited most of the frameworks from [TextZoom](https://github.com/JasonBoy1/TextZoom) and use the pretrained CRNN model from [CRNN](https://github.com/meijieru/crnn.pytorch).
Thanks for your contribution! 

[@JasonBoy1](https://github.com/JasonBoy1) 

 [@meijieru](https://github.com/meijieru)


## Citation
```python
@inproceedings{chen2021scene,
  title={Scene Text Telescope: Text-Focused Scene Image Super-Resolution},
  author={Chen, Jingye and Li, Bin and Xue, Xiangyang},
  booktitle={CVPR},
  pages={12026--12035},
  year={2021}
}
```