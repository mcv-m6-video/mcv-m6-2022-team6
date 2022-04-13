import torch
import torch.nn as nn
import torchvision.models as models


class Siamese(nn.Module):

	def __init__(self):
		super(Siamese, self).__init__()
		self.cnn1 = nn.Sequential(
			nn.ReflectionPad2d(1),
			nn.Conv2d(3, 4, kernel_size=3),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(4),
			nn.Dropout2d(p=.2),

			nn.ReflectionPad2d(1),
			nn.Conv2d(4, 8, kernel_size=3),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(8),
			nn.Dropout2d(p=.2),

			nn.ReflectionPad2d(1),
			nn.Conv2d(8, 8, kernel_size=3),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(8),
			nn.Dropout2d(p=.2),
		)

		self.fc1 = nn.Sequential(
			nn.Linear(8 * 100 * 100, 500),
			nn.ReLU(inplace=True),

			nn.Linear(500, 500),
			nn.ReLU(inplace=True),

			nn.Linear(500, 5)
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
