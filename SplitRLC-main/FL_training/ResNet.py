import torch.nn as nn

import glinear
import glinear
# Build the VGG model according to location and split_layer

# 3x3 卷积定义
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResNet(nn.Module):
	def __init__(self, location, vgg_name, split_layer, cfg, block, layers):
		super(ResNet, self).__init__()
		assert split_layer < len(cfg[vgg_name])
		self.location = location
		self.in_channels = 16
		self.conv, self.bn, self.relu, self.layer1, self.layer2, self.layer3, self.avg_pool, self.fc = self._make_layers(
			cfg[vgg_name], block, layers)

	def make_layer(self, block, out_channels, blocks, stride=1):
		downsample = None
		if (stride != 1) or (self.in_channels != out_channels):
			downsample = nn.Sequential(
				conv3x3(self.in_channels, out_channels, stride=stride),
				nn.BatchNorm2d(out_channels))
		layers = []
		layers.append(block(self.in_channels, out_channels, stride, downsample))
		self.in_channels = out_channels
		for i in range(1, blocks):
			layers.append(block(out_channels, out_channels))
		return nn.Sequential(*layers)

	def forward(self, x):
		out = self.conv(x)
		out = self.bn(out)
		out = self.relu(out)
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.avg_pool(out)
		out = out.view(out.size(0), -1)
		out = self.fc(out)

		return out



	def _make_layers(self, cfg, block, layers):
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
			if x[0] == 'A':
				self.conv = conv3x3(3, 16)
			if x[0] == 'B':
				self.bn = nn.BatchNorm2d(16)
			if x[0] == 'C':
				self.relu = nn.ReLU(inplace=True)
			if x[0] == 'D':
				self.layer1 = self.make_layer(block, 16, layers[0])
			if x[0] == 'E':
				self.layer2 = self.make_layer(block, 32, layers[1], 2)
			if x[0] == 'F':
				self.layer3 = self.make_layer(block, 64, layers[2], 2)
			if x[0] == 'G':
				self.avg_pool = nn.AvgPool2d(8)
			if x[0] == 'H':
				self.fc = nn.Linear(64, 10)

		return self.conv, self.bn, self.relu, self.layer1, self.layer2, self.layer3, self.avg_pool, self.fc

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

