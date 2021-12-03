# Text Gestalt

This is the code for AAAI2022 paper "Text Gestalt: Stroke-Aware Scene Text Image Super-Resolution". *(In preparation for the camera ready version...)*

![architecture](./image/architecture.png)

## TODO List
* [ ] Publish pre-trained weights and logs
* [ ] Make this paper publicly available

## Dependencies
Build up an environment with python3.6, and download corresponding libraries with pip
```python
pip install -r requirement.txt
```

## Pre-trained Model
Will be available soon

## Dataset
Download all resources at [BaiduYunDisk](https://pan.baidu.com/s/1MeFKnF5tWiL7ts00SHLM2A) with password: mteb

All the resources shoulded be placed under ```./dataset/mydata```, for example
```python
./dataset/mydata/train1
./dataset/mydata/train2
./dataset/mydata/test
./dataset/mydata/english_decomposition.txt
./dataset/mydata/crnn.pth
./dataset/mydata/pretrain_transformer_stroke_decomposition.pth
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

## Acknowledgement
We inherited most of the frameworks from [TextZoom](https://github.com/JasonBoy1/TextZoom) and use the pretrained CRNN model from [CRNN](https://github.com/meijieru/crnn.pytorch).
Thanks for your contribution! 

[@JasonBoy1](https://github.com/JasonBoy1) 

[@meijieru](https://github.com/meijieru)


## Citation
Will be available soon
