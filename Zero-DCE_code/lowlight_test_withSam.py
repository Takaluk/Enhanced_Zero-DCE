import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from PIL import Image
import glob
import time
import Myloss
import light_curve
import cv2
import matplotlib.pyplot as plt
import segmentation

 
def lowlight(image_path, DCE_net, sam):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	start = time.time()

	image_bgr = cv2.imread(image_path)
	data_lowlight = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

	loss_spa = loss_col = loss_exp = loss_cont = loss_lp = loss_TV = 0
	mask_imgs = sam.mk_mask(data_lowlight)
	result_imgs = []

	for (img, mask) in mask_imgs:
		# light_curve.plot_brightness(image_path)
		img = (np.asarray(img)/255.0)
		img = torch.from_numpy(img).float()
		img = img.permute(2,0,1)
		img = img.cuda().unsqueeze(0)

		_,enhanced_image,A = DCE_net(img)

		L_color = Myloss.L_color()
		L_spa = Myloss.L_spa()
		L_exp = Myloss.L_exp(16,0.6)
		L_TV = Myloss.L_TV()
		L_cont = 	Myloss.ContrastLoss()
		L_lp = 	Myloss.LPIPSloss()

		loss_col += torch.mean(L_color(enhanced_image))
		loss_spa += torch.mean(L_spa(enhanced_image, img))
		loss_exp += torch.mean(L_exp(enhanced_image))
		Loss_TV = 200*L_TV(A)
		loss_cont += torch.mean(L_cont(enhanced_image, img))
		loss_lp += torch.mean(L_lp(enhanced_image, img))


		result = torchvision.utils.make_grid(enhanced_image)
		result = result.detach().cpu().numpy()
		result = np.transpose(result, (1, 2, 0))
		result_imgs.append((result, mask))

	(enhanced_image, mask_remain) = result_imgs[-1]
	for (img, mask) in result_imgs[:-1]:
		enhanced_image[mask==True] = img[mask==True]
		mask_remain = mask_remain & mask

	end_time = (time.time() - start)
	print(end_time)
	print("loss_spa: "+  str(loss_spa / len(mask_imgs)))
	print("loss_col: "+  str(loss_col / len(mask_imgs)))
	print("loss_exp: "+  str(loss_exp / len(mask_imgs)))
	print("loss_TV: "+  str(Loss_TV / len(mask_imgs)))
	print("loss_cont: "+  str(loss_cont / len(mask_imgs)))
	print("loss_lp: "+  str(loss_lp / len(mask_imgs)))

	image_path = image_path.replace('test_data','result')
	result_path = image_path.replace('.jpg', 'SAM.jpg')
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

	enhanced_image = cv2.normalize(enhanced_image, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	cv2.imwrite(result_path, cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))

	if config.show_result:
		plt.imshow(enhanced_image)
		plt.axis('off')
		plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--imgpath', type=str, default= 'data/test_data')
	parser.add_argument('--weight', type=int, default= 10)
	parser.add_argument('--checkpoint', type=str ,default="weights/sam_vit_h_4b8939.pth")
	parser.add_argument('--show_result', type=bool ,default=False)
	config = parser.parse_args()
  
	DCE_net = model.enhance_net_nopool().cuda()
	DCE_net.load_state_dict(torch.load('snapshots/Epoch'+ str(config.weight) +'.pth'))
	sam = segmentation.SAM(config.checkpoint)

# RUN
	for image in glob.glob(config.imgpath + '/*'):
		print(image)
		lowlight(image, DCE_net, sam)
