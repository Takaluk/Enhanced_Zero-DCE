import os
import sys
import argparse
import glob
import cv2
import supervision as sv
import torch
import numpy as np

def mk_mask(config):
  DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  MODEL_TYPE = "vit_h"

  from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

  sam = sam_model_registry[MODEL_TYPE](checkpoint= config.CHECKPOINT_PATH).to(device=DEVICE)
  mask_generator = SamAutomaticMaskGenerator(sam)

  train_images_path = "data/train_data/"
  train_dataset = glob.glob(train_images_path + "*.jpg")

  for IMAGE_PATH in train_dataset:
    IMAGE_NAME = os.path.split(IMAGE_PATH)[-1]
    if not os.path.exists("train_masks/"+ IMAGE_NAME + ".npz"):

      image_bgr = cv2.imread(IMAGE_PATH)
      image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
      sam_result = mask_generator.generate(image_rgb)

      masks = [
        mask['segmentation']
        for mask
        in sorted(sam_result, key=lambda x: x['area'], reverse=True)]
      mask_path = "train_masks/" + IMAGE_NAME
      np.savez_compressed(mask_path,masks)
    else:
      print("pass : " + IMAGE_NAME)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument('--CHECKPOINT_PATH', type=str ,default="/content/drive/MyDrive/Zero-DCE-master/Zero-DCE_code/weights/sam_vit_h_4b8939.pth")
  config = parser.parse_args()

  if not os.path.exists("train_masks/"):
    os.mkdir("train_masks/")

  mk_mask(config)