# LCDBNet

Division Gets Better: Learning Brightness-Aware and Detail-Sensitive Representations for Low-Light Image Enhancement.

Our paper has been accepted by KBS. 

Our pretrained model is coming soon.

## Environment
* Python
* Pytorch
* numpy
* tqdm
* pandas


## Train
Download LOL dataset from (https://daooshee.github.io/BMVC2018website/) and change the dataset path in 'training.yaml', and then

```
python train_denoise.py
```


## Test
test LOL dataset. Change the testdataset path and then

```
python test.py
```

or unpair dataset

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




