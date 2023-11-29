import argparse
import glob
import os
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim
import torchvision
import wget

import light_curve
import model
import Myloss
import segmentation

warnings.filterwarnings("ignore", category=UserWarning) 
 
def lowlight(config):
	img_folder_path = config.src_img_path
	result_folder_path = config.dst_img_path
	if img_folder_path[-1] == '/':
		img_folder_path = img_folder_path[:len(img_folder_path)-1]
	if result_folder_path is not None and result_folder_path[-1] == '/':
		result_folder_path = result_folder_path[:len(result_folder_path)-1]

	if config.device == 'cuda':
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		if torch.cuda.is_available():
			os.environ['CUDA_VISIBLE_DEVICES']='0'
		else:
			print('No CUDA GPU found! Using CPU...')
	else:
		device = torch.device('cpu')

	DCE_net = model.enhance_net_nopool().to(device)
	DCE_net.load_state_dict(torch.load('snapshots/Epoch'+ str(config.weight) +'.pth', map_location= device))
	if config.with_sam:
		if os.path.isfile(config.checkpoint):
			sam = segmentation.SAM(config.checkpoint, device)
		else:
			print("SAM checkpoint not found! Downloading default checkpoint...")
			os.makedirs('weights', exist_ok= True)
			wget.download('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth', out= 'weights/')
			sam = segmentation.SAM('weights/sam_vit_h_4b8939.pth', device)

	L_color	= Myloss.L_color()
	L_spa	= Myloss.L_spa(device)
	L_exp 	= Myloss.L_exp(16,0.6, device)
	L_TV	= Myloss.L_TV()
	L_lp	= Myloss.LPIPSloss(device)
	# L_cont		= 	Myloss.ContrastLoss()

	for image_path in glob.glob(config.src_img_path + '/**/*', recursive= True):
		if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
			continue
		print('\nWorking on image ' + image_path )
		
		image_bgr = cv2.imread(image_path)
		data_lowlight = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
		# light_curve.plot_brightness(image_path)

		loss_spa = loss_col = loss_exp = loss_cont = loss_lp = loss_TV = 0

		if config.with_sam:
			mask_imgs = sam.mk_mask(data_lowlight)
		else:
			height, width, channel = data_lowlight.shape
			mask_imgs = [(data_lowlight, np.ones((height, width), dtype=bool))]
		result_imgs = []
		num_mask = len(mask_imgs)

		for (img, mask) in mask_imgs:
			img = (np.asarray(img)/255.0)
			img = torch.from_numpy(img).float()
			img = img.permute(2,0,1)
			img = img.to(device).unsqueeze(0)

			_, enhanced_image, A = DCE_net(img)

			loss_col += torch.mean(L_color(enhanced_image)).item()
			loss_spa += torch.mean(L_spa(enhanced_image, img)).item()
			loss_exp += torch.mean(L_exp(enhanced_image)).item()
			loss_TV = 200*L_TV(A).item()
			loss_lp += torch.mean(L_lp(enhanced_image, img)).item()
			# loss_cont += torch.mean(L_cont(enhanced_image, img))

			result = torchvision.utils.make_grid(enhanced_image)
			result = result.detach().cpu().numpy()
			result = np.transpose(result, (1, 2, 0))
			result_imgs.append((result, mask))

		(enhanced_image, mask_remain) = result_imgs[-1]
		for (img, mask) in result_imgs[:-1]:
			enhanced_image[mask==True] = img[mask==True]
			mask_remain = mask_remain & mask

		loss_spa /= num_mask
		loss_col /= num_mask
		loss_exp /= num_mask
		loss_TV /= num_mask
		loss_lp /= num_mask
		loss_cont /= num_mask

		print("loss_spa: "+  "{:.5f}".format(loss_spa), end='	')
		print("loss_col: "+  "{:.5f}".format(loss_col), end='	')
		print("loss_exp: "+  "{:.5f}".format(loss_exp), end='	')
		print("loss_TV: "+  "{:.5f}".format(loss_TV), end='	')
		print("loss_lp: "+  "{:.5f}".format(loss_lp))
		# print("loss_cont: "+  "{:.5f}".format(loss_cont), end='	')

		print("Loss: " + "{:.10f}".format(loss_spa+loss_col+loss_exp+loss_TV+loss_lp))

		result_path = image_path.replace(img_folder_path, img_folder_path + '_result')if result_folder_path is None else image_path.replace(img_folder_path, result_folder_path)
		# if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs('/'.join(result_path.split('/')[:-1]), exist_ok= True)

		enhanced_image = cv2.normalize(enhanced_image, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
		cv2.imwrite(result_path, cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))

		if config.show_result:
			plt.imshow(enhanced_image)
			plt.axis('off')
			plt.show()
		if config.show_plot:
			light_curve.plot_brightness(result_path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--weight', type=int, default=888)
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--with_sam', action='store_true')
	parser.add_argument('--show_plot', type=bool ,default=False)
	parser.add_argument('--show_result', type=bool ,default=False)
	parser.add_argument('--src_img_path', type=str, default='data/test_data')
	parser.add_argument('--dst_img_path', type=str, default=None)
	parser.add_argument('--checkpoint', type=str ,default="weights/sam_vit_h_4b8939.pth")
	
	config = parser.parse_args()
  
	lowlight(config)

