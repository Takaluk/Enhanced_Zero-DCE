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
import Myloss
import numpy as np
from torchvision import transforms


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)


def train(config):
	img_folder_path = config.lowlight_images_path
	if img_folder_path[-1] == '/':
		img_folder_path = img_folder_path[:len(img_folder_path)-1]

	if config.device == 'cuda':
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		if torch.cuda.is_available():
			os.environ['CUDA_VISIBLE_DEVICES']='0'
		else:
			print('No CUDA GPU found! Using CPU...')
	else:
		device = torch.device('cpu')

	DCE_net = model.enhance_net_nopool().to(device)

	DCE_net.apply(weights_init)
	if config.load_pretrain == True:
		DCE_net.load_state_dict(torch.load(config.pretrain_dir, map_location= device))
	train_dataset = dataloader.lowlight_loader(img_folder_path)		
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

	L_color = Myloss.L_color()
	L_spa = Myloss.L_spa(device)
	L_exp = Myloss.L_exp(16,0.6, device)
	L_TV = Myloss.L_TV()
	L_lp = Myloss.LPIPSloss(device)
	# L_cont = 	Myloss.ContrastLoss()

	optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
	DCE_net.train()

	for epoch in range(config.num_epochs):
		for iteration, img_lowlight in enumerate(train_loader):

			img_lowlight = img_lowlight.to(device)

			enhanced_image_1,enhanced_image,A  = DCE_net(img_lowlight)

			Loss_TV = 200*L_TV(A)
			loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
			loss_col = 5*torch.mean(L_color(enhanced_image))
			loss_exp = 10*torch.mean(L_exp(enhanced_image))
			loss_lp = 3*torch.mean(L_lp(enhanced_image, img_lowlight))
			#loss_cont = 2*torch.mean(L_cont(enhanced_image, img_lowlight))

			# best_loss
			loss =  Loss_TV + loss_spa + loss_col + loss_exp + loss_lp

			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(DCE_net.parameters(),config.grad_clip_norm)
			optimizer.step()

			if ((iteration+1) % config.display_iter) == 0:
				print("Loss at iteration", iteration+1, ":", loss.item())
			if ((iteration+1) % config.snapshot_iter) == 0:
				torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth') 		




if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--lowlight_images_path', type=str, default="data/test_data/DICM")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=1)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots/Epoch999.pth")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)

	train(config)








	