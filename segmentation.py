import argparse
import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np
import supervision as sv
import torch

numpy.set_printoptions()

def show_anns(anns):
	if len(anns) == 0:
			return
	sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
	ax = plt.gca()
	ax.set_autoscale_on(False)

	img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
	img[:,:,3] = 0
	for ann in sorted_anns:
			m = ann['segmentation']
			color_mask = np.concatenate([np.random.random(3), [0.35]])
			img[m] = color_mask
	# ax.imshow(img)

class SAM:
	def __init__(self, checkpoint, device):
		self.checkpoint = checkpoint

		self.DEVICE = device
		self.MODEL_TYPE = "vit_h"

		from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
		                              sam_model_registry)

		sam = sam_model_registry[self.MODEL_TYPE](self.checkpoint).to(self.DEVICE)
		self.mask_generator = SamAutomaticMaskGenerator(sam)

	def mk_mask(self, image):
		print("Masking..")
		sam_result = self.mask_generator.generate(image)

		pixels = image.shape[0] * image.shape[1]

		sam_result[:] = [d for d in sam_result if not int(d.get('area')) < pixels/10]
		masks = [
			mask['segmentation']
			for mask
			in sorted(sam_result, key=lambda x: x['area'], reverse=True)
			]

		mask_remain = np.full((image.shape[0], image.shape[1]), False)
		mask_list = []

		for mask in masks:
			img_copy = image.copy();
			mask_pixels = img_copy[mask==True]
			avg_color = np.mean(mask_pixels, axis = 0)
			img_copy[mask==False] = avg_color

			mask_list.append((img_copy, mask))

			mask_remain = mask_remain | mask

		mask_remain = np.logical_not(mask_remain)
		img_remain = image.copy()
		mask_pixels = img_remain[mask_remain==True]
		avg_color = np.mean(mask_pixels, axis = 0)
		img_remain[mask_remain==False] = avg_color

		mask_list.append((img_remain, mask_remain))

		return mask_list
