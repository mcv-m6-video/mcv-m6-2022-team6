import torch
import torch.nn as nn
import torchvision.models as models


class Siamese(nn.Module):

	def __init__(self):
		super(Siamese, self).__init__()
		self.cnn1 = nn.Sequential(
			nn.Conv2d(3, 96, kernel_size=11, stride=1),
			nn.ReLU(inplace=True),
			nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
			nn.MaxPool2d(3, stride=2),

			nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
			nn.ReLU(inplace=True),
			nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
			nn.MaxPool2d(3, stride=2),
			nn.Dropout2d(p=0.3),

			nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),

			nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(3, stride=2),
			nn.Dropout2d(p=0.3),
		)

		# Defining the fully connected layers
		self.fc1 = nn.Sequential(
			nn.Linear(6400, 1024),
			nn.ReLU(inplace=True),
			nn.Dropout2d(p=0.5),

			nn.Linear(1024, 128),
			nn.ReLU(inplace=True),

			nn.Linear(128, 2)
		)

	def forward_one(self, x):
		x = self.cnn1(x)
		x = x.view(x.size()[0], -1)
		x = self.fc1(x)
		return x

	def forward(self, x1, x2):
		out1 = self.forward_one(x1)
		out2 = self.forward_one(x2)
		#  return self.sigmoid(out)
		return out1, out2


# for test
if __name__ == '__main__':
	net = Siamese()
	print(net)
	print(list(net.parameters()))
