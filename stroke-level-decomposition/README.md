# Stroke Level Decomposition

This is the code for our IJCAI2021 paper "Zero-Shot Chinese Character Recognition with Stroke-Level Decomposition". [[link]](https://github.com/FudanVI/FudanOCR/tree/main/stroke-level-decomposition/document)

![architecture](./image/architecture.png)


## Dependencies
Build up an environment with python3.6, and download corresponding libraries with pip
```python
pip install -r requirement.txt
```

## Dataset
Download all resources at [BaiduYunDisk]() with password: ????, or [Dropbox]()

Dataset for seen settign:
* HWDB1.0 handwritten dataset 
* HWDB1.1 handwritten dataset 
* ICDAR2013 handwritten dataset
* CTW scene character dataset

Dataset for zero-shot experiments:
* Handwritten dataset in **Character Zero-Shot** Setting
* Handwritten dataset in **Radical Zero-Shot** Setting
* Printed artistic dataset  in **Character Zero-Shot** Setting
* Printed artistic dataset in **Radical Zero-Shot** Setting
* CTW scene character dataset  in **Character Zero-Shot** Setting
* CTW scene character dataset in **Radical Zero-Shot** Setting

All the resources shoulded be placed under ```./dataset/mydata```, for example
```python
./dataset/mydata//train_1000
./dataset/mydata//test_1000
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