# Enhanced_Zero-DCE
Enhanced Zero-DCE is CSE graduation project focused on augmenting the performance and quality of the original Zero-DCE model.

## Original Zero-DCE
[original project page](https://github.com/Li-Chongyi/Zero-DCE)

## Enhancement tasks
[Tasks page](https://crawling-hugger-363.notion.site/Enhanced_zero-dce-acacc4c4196f499298f3c5ef18b38b9c?pvs=4)

## Requirements
### Install all requirements
```
pip install -r requirements.txt
```

#### Zero-DCE
The basics are same as [original Zero-DCE project](https://github.com/Li-Chongyi/Zero-DCE)
1. Python 3.7
2. Pytorch 1.0.0
3. opencv
4. torchvision 0.2.1
5. cuda 10.0


#### LPIPS
6. lpips

#### SAM
[SAM project page](https://github.com/facebookresearch/segment-anything)

7. segment-anything
8. jupyter_bbox_widget
9. roboflow
10. dataclasses-json 
11. supervision

## Train
```
python train.py
```
## Test
```
python infer.py
```
