
import torch.nn as nn

def make_layers(cfg, batch_norm=False, dropout=0):
	layers = []
	in_channels = 3
	for i, v in enumerate(cfg):
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			if isinstance(v, (tuple, list)):
				out_channels, kernel_size, padding = v
			else:
				out_channels, kernel_size, padding = v, 3, 1
			conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
			layers = [conv2d]
			if batch_norm:
				layers.append(nn.BatchNorm2d(out_channels, affine=False))
			layers.append(nn.ReLU())
			if dropout > 0 and dropout < 1:
				layers.append(nn.Dropout(dropout))
			in_channels = out_channels
	return nn.Sequential(*layers)


class CIFAR(nn.Module):

	def __init__(self, num_classes=10):
		nn.Module.__init__(self)
		n_channel = 128
		cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 3, 0), 'M']
		self.features = make_layers(cfg, batch_norm=True)
		self.classifier = nn.Sequential(
			nn.Linear(8 * n_channel, num_classes)
		)
		# print(self.features)
		# print(self.classifier)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

class LocConvNet(nn.Module):

	def __init__(self, args):
		nn.Module.__init__(self)
		# The size of the input image is clipped to (3, 256, 256)
		features = 128
		cfg = [features, 'M'] * 8
		self.features = make_layers(cfg, args.batch_norm, args.dropout)
		self.classifier = nn.Sequential(
			nn.Linear(features, args.num_classes)
		)
		# print(self.features)
		# print(self.classifier)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x
		