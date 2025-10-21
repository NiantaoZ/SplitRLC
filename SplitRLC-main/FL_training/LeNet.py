import torch.nn as nn
import torch.nn.functional as F
import glinear
import glinear
# Build the VGG model according to location and split_layer
class LeNet(nn.Module):
	def __init__(self, location, vgg_name, split_layer, cfg):
		super(LeNet, self).__init__()
		assert split_layer < len(cfg[vgg_name])
		self.split_layer = split_layer
		self.location = location
		self.features, self.fc1, self.fc2, self.fc3 = self._make_layers(cfg[vgg_name])
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
		out = out.view(-1 , 16*5*5)
		out = F.relu(self.fc1(out))
		out = F.relu(self.fc2(out))
		out = self.fc3(out)
		output = F.log_softmax(out, dim=1)

		return output

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
			if x[0] == 'B':
				features += [nn.AvgPool2d(kernel_size=kernel_size, stride=2)]

			if x[0] == 'A':
				features += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),nn.ReLU()]

			if x[0] == 'C':
				self.fc1 = nn.Linear(in_channels,out_channels)

			if x[0] == 'D':
				self.fc2 = nn.Linear(in_channels,out_channels)

			if x[0] == 'E':
				self.fc3 = nn.Linear(in_channels,out_channels)
		return nn.Sequential(*features),self.fc1, self.fc2, self.fc3

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

