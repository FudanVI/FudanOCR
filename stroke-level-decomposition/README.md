# Stroke Level Decomposition

This is the code for our IJCAI2021 paper "Zero-Shot Chinese Character Recognition with Stroke-Level Decomposition". [[link]](https://github.com/FudanVI/FudanOCR/tree/main/stroke-level-decomposition/document)

![architecture](./image/architecture.png)


## Dependencies
Build up an environment with python3.6, and download corresponding libraries with pip
```python
pip install -r requirement.txt
```

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


## Experiment
Please remember to modify ```config.py``` and then execute
```python
CUDA_VISIBLE_DEVICES=GPU_NUM python train.py
```



## Citation
```python
@inproceedings{not available now,
  title={Zero-Shot Chinese Character Recognition with Stroke-Level Decomposition},
  author={Jingye Chen and Bin Li and Xiangyang Xue},
  booktitle={IJCAI},
  year={2021},
}
```