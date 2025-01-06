import sys

# Network configration
SERVER_ADDR= '192.168.110.35'
SERVER_PORT = 51000

K = 5# Number of devices
G = 3 # Number of groups

# Unique clients order
HOST2IP = {'splitrlc_node1':'192.168.110.30', 'splitrlc_node2':'192.168.110.31' , 'splitrlc_node3':'192.168.110.32', 'splitrlc_node4':'192.168.110.33','splitrlc_node5':'192.168.110.34' }
CLIENTS_CONFIG= {'192.168.110.30':0, '192.168.110.31':1, '192.168.110.32':2, '192.168.110.33':3,'192.168.110.34':4}
CLIENTS_LIST= [ '192.168.110.30', '192.168.110.31', '192.168.110.32', '192.168.110.33','192.168.110.34']

# Dataset configration
dataset_name = 'CIFAR10'
home = sys.path[0].split('SplitRLC')[0] + 'SplitRLC'
dataset_path = './dataset/'+ dataset_name +'/'
N = 50000 # data length


# Model configration
model_cfg = {
	# (Type, in_channels, out_channels, kernel_size, out_size(c_out*h*w), flops(c_out*h*w*k*k*c_in))
	'VGG5' : [('C', 3, 32, 3, 32*32*32, 32*32*32*3*3*3), ('M', 32, 32, 2, 32*16*16, 0),
	('C', 32, 64, 3, 64*16*16, 64*16*16*3*3*32), ('M', 64, 64, 2, 64*8*8, 0),
	('C', 64, 64, 3, 64*8*8, 64*8*8*3*3*64),
	('D', 8*8*64, 128, 1, 64, 128*8*8*64),
	('D', 128, 10, 1, 10, 128*10)]

	#'AlexNet': [('C', 3, 96, 11, 32*32*32, 32*32*32*3*3*3), ('M', 96, 96, 3, 32*16*16, 0),
	#('C', 96, 256, 5, 64*16*16, 64*16*16*3*3*32), ('M', 256,256, 3, 64*8*8, 0),
	#('C', 256, 384, 3, 64*8*8, 64*8*8*3*3*64),
	#('C', 384, 384, 3, 64*8*8, 64*8*8*3*3*64),
	#('C', 384, 256, 3, 64*8*8, 64*8*8*3*3*64),
	#('M', 256,256,3, 64*8*8,0),
	#('E', 256,512, 0, 0, 0),
	#('G', 512, 64, 1, 64, 128*64),
	#('D', 64, 10, 1, 64, 128*64)
	#]
	# 'ResNet': [('A', 3, 96, 11, 32*32*32, 32*32*32*3*3*3), ('B', 96, 96, 3, 32*16*16, 0),
	# ('C', 96, 256, 5, 64*16*16, 64*16*16*3*3*32), ('D', 256, 256, 3, 64*8*8, 0),
	# ('E', 256, 384, 3, 64*8*8, 64*8*8*3*3*64),
	# ('F', 384, 384, 3, 64*8*8, 64*8*8*3*3*64),
	# ('G', 384, 256, 3, 64*8*8, 64*8*8*3*3*64),
	# ('H', 256, 256, 3, 64*8*8, 0)]
	# # #
	# 'LeNet': [('A', 3, 6, 5, 32*32*32, 32*32*32*3*3*3), ('B', 6, 6, 2, 32*16*16, 0),
	# ('A', 6, 16, 5, 64*16*16, 64*16*16*3*3*32), ('B', 16, 16, 2, 64*8*8, 0),
	# ('C', 16*5*5, 1024, 3, 64*8*8, 64*8*8*3*3*64),
	# ('D', 1024, 128, 3, 64*8*8, 64*8*8*3*3*64),
	# ('E', 128, 10, 3, 64*8*8, 64*8*8*3*3*64)]
}
model_name = 'VGG5'
model_size = 1.28
model_flops = 32.902
total_flops = 8488192
split_layer = [6,6,6,6,6] #Initial split layers
model_len =7


# FL training configration
R = 100 # FL rounds
LR = 0.01 # Learning rate
B = 100 # Batch size


# RL training configration
max_episodes = 100         # max training episodes
max_timesteps = 100        # max timesteps in one episode
exploration_times = 20	   # exploration times without std decay
n_latent_var = 64          # number of variables in hidden layer
action_std = 0.5           # constant std for action distribution (Multivariate Normal)
update_timestep = 10       # update policy every n timesteps
K_epochs = 50              # update policy for K epochs
eps_clip = 0.2             # clip parameter for PPO
rl_gamma = 0.9             # discount factor
rl_b = 100				   # Batchsize
rl_lr = 0.0003             # parameters for Adam optimizer
rl_betas = (0.9, 0.999)
iteration = {'192.168.110.30' : 5, '192.168.110.31': 10, '192.168.110.32': 5, '192.168.110.33': 5,'192.168.110.34' : 5}  # infer times for each device

random = True
random_seed = 0
