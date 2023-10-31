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
 
def lowlight(image_path, DCE_net):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	data_lowlight = Image.open(image_path)

	data_lowlight.show()
	light_curve.plot_brightness(image_path)
	data_lowlight = (np.asarray(data_lowlight)/255.0)
      
	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	start = time.time()
	_,enhanced_image,A = DCE_net(data_lowlight)

	L_color = Myloss.L_color()
	L_spa = Myloss.L_spa()
	L_exp = Myloss.L_exp(16,0.6)
	L_TV = Myloss.L_TV()
	Loss_TV = 200*L_TV(A)
	L_cont = 	Myloss.ContrastLoss()
	L_lp = 	Myloss.LPIPSloss()
	#L_dr = 	Myloss.DramaticLoss(0,150)

	loss_spa = torch.mean(L_spa(enhanced_image, data_lowlight))
	loss_col = torch.mean(L_color(enhanced_image))
	loss_exp = torch.mean(L_exp(enhanced_image))
	loss_cont = torch.mean(L_cont(enhanced_image, data_lowlight))
	loss_lp = torch.mean(L_lp(enhanced_image, data_lowlight))
	#loss_dr = 5*torch.mean(L_dr(enhanced_image, data_lowlight))
	print("TV: "+  str(Loss_TV))
	print("loss_spa: "+  str(loss_spa))
	print("loss_col: "+  str(loss_col))
	print("loss_exp: "+  str(loss_exp))
	print("loss_cont: "+  str(loss_cont))
	print("loss_lp: "+  str(loss_lp))
	#print(loss_dr)
	end_time = (time.time() - start)
	#print(end_time)#
	image_path = image_path.replace('test_data','result')
	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

	torchvision.utils.save_image(enhanced_image, result_path)

	if config.show_result:
		after = Image.open(result_path)
		after.show()
	if config.show_plot:
		light_curve.plot_brightness(result_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--imgpath', type=str, default= 'data/test_data')
	parser.add_argument('--weight', type=int, default= 10)
	parser.add_argument('--show_result', type=bool ,default=False)
	parser.add_argument('--show_plot', type=bool ,default=False)
	config = parser.parse_args()
  
	DCE_net = model.enhance_net_nopool().cuda()
	DCE_net.load_state_dict(torch.load('snapshots/Epoch'+ str(config.weight) +'.pth'))

# RUN
	for image in glob.glob(config.imgpath + '/*'):
		print(image)
		lowlight(image, DCE_net)



# if __name__ == '__main__':
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('--weight', type=int, default=10)
# 	config = parser.parse_args()
# # test_images
# 	with torch.no_grad():
# 		filePath = 'data/test_data/'
	
# 		file_list = os.listdir(filePath)

# 		for file_name in file_list:
# 			test_list = glob.glob(filePath+file_name+"/*") 
# 			for image in test_list:
# 				# image = image
# 				print(image)
# 				lowlight(image)
