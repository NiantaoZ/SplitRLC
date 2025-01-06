import torch.nn as nn

import glinear
import glinear
# Build the VGG model according to location and split_layer
class AlexNet(nn.Module):
	def __init__(self, location, vgg_name, split_layer, cfg):
		super(AlexNet, self).__init__()
		assert split_layer < len(cfg[vgg_name])
		self.split_layer = split_layer
		self.location = location
		self.features, self.pro, self.out, self.L2 = self._make_layers(cfg[vgg_name])
		self._initialize_weights()
		self.Flatten = nn.Flatten(),
		self.dropout = nn.Dropout(p=0.5)
		self.bn1 = nn.BatchNorm1d(128)
		self.bn2 = nn.BatchNorm1d(64)

	def forward(self, x):
		if len(self.features) > 0:
			out = self.features(x)
		else:
			out = x
		# if len(self.denses) > 0:
		# 	out = out.view(out.size(0), -1)
		# 	out = self.denses(out)
		# else:
		# 	out = out
		out = out.view(out.size(0), -1)
		pro = self.pro(out)
		L2 = self.L2(pro)
		L2 = self.dropout(L2)
		out = self.out(L2)

		return out

	def _make_layers(self, cfg):
		features = []
		denses = []
		if self.location == 'Server':
			cfg = cfg[self.split_layer+1 :]

		if self.location == 'Client':
			cfg = cfg[:self.split_layer+1]

		if self.location == 'Unit': # Get the holistic model（获得整体模型）
			pass

		for x in cfg:
			in_channels, out_channels = x[1], x[2]
			kernel_size = x[3]
			if x[0] == 'M':
				features += [nn.MaxPool2d(kernel_size=kernel_size, stride=2)]
			# if x[0] == 'F':
			# 	self.L1 = nn.Linear(in_channels,out_channels)
			if x[0] == 'G':
				self.L2 = nn.Linear(in_channels,out_channels)
			if x[0] == 'E':
				self.pro =nn.Linear(in_channels,out_channels)
			if x[0] == 'D':
				self.out = nn.Linear(in_channels,out_channels)

			if x[0] == 'C':
				features += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
						   nn.ReLU(inplace=True)]

		return nn.Sequential(*features), self.pro, self.out,  self.L2

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

