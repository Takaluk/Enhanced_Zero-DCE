# Enhanced_Zero-DCE
Enhanced Zero-DCE is CSE graduation project focused on augmenting the performance and quality of the original Zero-DCE model.

## Original Zero-DCE
[original project page](https://github.com/Li-Chongyi/Zero-DCE)

[training data](https://drive.google.com/file/d/1GAB3uGsmAyLgtDBDONbil08vVu5wJcG3/view)

## Enhancement tasks
[Tasks page](https://crawling-hugger-363.notion.site/Enhanced_zero-dce-acacc4c4196f499298f3c5ef18b38b9c?pvs=4)

## Requirements
The basics are same as [original Zero-DCE project](https://github.com/Li-Chongyi/Zero-DCE)

### LPIPS
```
pip install lpips
```
### SAM
[SAM project page](https://github.com/facebookresearch/segment-anything)

cd Zero-DCE_code
```
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install -q jupyter_bbox_widget roboflow dataclasses-json supervision
mkdir weights
cd weights
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
### Train
cd Zero-DCE_code
```
python lowlight_train.py
```
### Test
cd Zero-DCE_code
```
python lowlight_test.py
```
