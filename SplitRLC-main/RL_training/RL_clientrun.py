import torch
import socket
import multiprocessing
import numpy as np

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('../')
from RLEnv import RL_Client
import config
import utils
import random
import logging
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

if config.random:
	torch.manual_seed(config.random_seed)
	np.random.seed(config.random_seed)
	logger.info('Random seed: {}'.format(config.random_seed))

first = True
ip_address = config.HOST2IP[socket.gethostname()]
index = config.CLIENTS_CONFIG[ip_address]
datalen = config.N / config.K
split_layer = config.split_layer[index]

logger.info('==> Preparing Data..')
cpu_count = multiprocessing.cpu_count()
def get_local_dataloader(CLIENT_IDEX, cpu_count):
        dataset_name = 'CIFAR10'
        dataset_path = './dataset/'+ dataset_name +'/'
        indices = list(range(50000))
        random.shuffle(indices)
        part_tr = indices[0 : 1000]

        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
        trainset = torchvision.datasets.CIFAR10(
                root=dataset_path, train=True, download=True, transform=transform_train)
        subset = Subset(trainset, part_tr)
        trainloader = DataLoader(subset, batch_size=100, shuffle=True, num_workers=cpu_count)
        #trainloader = DataLoader(trainset, batch_size=B, shuffle=True, num_workers=0)
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
        return trainloader,classes
trainloader, classes= utils.get_local_dataloader(index, cpu_count)

logger.info('==> Preparing RL_Client..')
rl_client = RL_Client(index, ip_address, config.SERVER_ADDR, config.SERVER_PORT, datalen, config.model_name, split_layer, config.model_cfg)

while True:
	reset_flag = rl_client.recv_msg(rl_client.sock, 'RESET_FLAG')[1]
	if reset_flag:
		rl_client.initialize(len(config.model_cfg[config.model_name])-1)
	else:
		logger.info('==> Next Timestep..')
		config.split_layer = rl_client.recv_msg(rl_client.sock, 'SPLIT_LAYERS')[1]
		rl_client.reinitialize(config.split_layer[index])

	logger.info('==> Training Start..')
	if first:
		rl_client.infer(trainloader)
		rl_client.infer(trainloader)
		first = False
	else:
		rl_client.infer(trainloader)

	
	


