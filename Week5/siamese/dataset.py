import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

#Path for Colab
PATH = 'patches/images'

#Path Local
#PATH = '../patches/images/'

class CarDataset(Dataset):

	def __init__(self, dataPd, transform=None):
		super(CarDataset, self).__init__()
		np.random.seed(0)
		self.num_classes = 2
		self.csv_data = dataPd
		self.transform = transform

	def __len__(self):
		return self.csv_data.size

	def load_img(self, data_file):
		return Image.open(f"{PATH}/{data_file.FILENAME}")

	def __getitem__(self, index):

		data = self.csv_data
		idx1 = np.random.randint(0, self.csv_data.shape[0])
		image1 = self.load_img(data.iloc[idx1])

		# get image from same class
		if index % 2 == 1:
			label = 0
			data_else = data[(data.ID == data.iloc[idx1].ID) & (data.FILENAME != data.iloc[idx1].FILENAME)]
		# get image from different class
		else:
			label = 1
			data_else = data[data.ID != data.iloc[idx1].ID]

		idx2 = np.random.randint(0, data_else.shape[0])
		image2 = self.load_img(data_else.iloc[idx2])

		if self.transform:
			image1 = self.transform(image1)
			image2 = self.transform(image2)

		return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))
