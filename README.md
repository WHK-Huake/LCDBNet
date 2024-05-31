# LCDBNet

Division Gets Better: Learning Brightness-Aware and Detail-Sensitive Representations for Low-Light Image Enhancement.

Our paper has been accepted by KBS. 

## Environment
* Python
* Pytorch
* numpy
* tqdm
* pandas

## Trained models
You can download our trained models from [BaiduPan [code:dlau]](https://pan.baidu.com/share/init?surl=t7vNrOhC3syIWhRrl4r9Qg). 

Place `checkpoints` in 'LCDBNet/'


## Train
Download LOL dataset from (https://daooshee.github.io/BMVC2018website/) and change the dataset path in `training.yaml`, and then

```
python train_denoise.py
```


## Test
test LOL dataset. Change the testdataset path of `test.py` and then run

```
python test.py
```

or unpair dataset run

```
python test_unpair.py
```

## Citation
If you find the code helpful in your research or work, please cite the following paper:
```
@article{Wang2024LCDBNet,
  author       = {Huake Wang and
                  Xiaoyang Yan and
                  Xingsong Hou and
                  Junhui Li and
                  Yujie Dun and
                  Kaibing Zhang},
  title        = {Division gets better: Learning brightness-aware and detail-sensitive representations for low-light image enhancement},
  journal      = {Knowledge-Based Systems},
  volume       = {},
  pages        = {111958},
  year         = {2024}
}
```




