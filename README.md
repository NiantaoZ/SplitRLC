## SplitRLC: Adaptive model Splitting with Sample-Efficient Reinforcement learning for Federated Split Learning

### About the research

In this study, we develop an effective FSL framework called SplitRLC to mitigate the impact of computing heterogeneity and adapt to different network bandwidth techniques. It enables flexible partitioning and distributed computing of DNN models in dynamic IoT environments, enhancing FL training efficiency.


![FSL](https://github.com/user-attachments/assets/cebf9947-6501-40ca-8c08-ce2656be5623)



To accelerate the training process of federated learning, SplitRLC uses a novel RL method to dynamically determine which layer of deep neural network (DNN) should be offloaded to each device on the server. Specifically, we propose SplitRLC, a novel RL-based adaptive splitting scheme that generates optimal SPs for the DNNs of each device in resource-constrained real-world FL settings. This design operates without relying on real supervised data and is capable of generating diverse virtual state-action pairs to enhance sample efficiency. Specifically, in RL we predict future states using a transition model, which is trained on real state-action pairs to forecast future states based on current states and actions. We then employ a backward transition model to predict previous states, forming a trajectory cycle. Building on this, numerous virtual state-action pairs can be generated for training state representations, with correctness ensured by cycle consistency constraints. Furthermore, the reliability-aware filtering mechanism utilizes ensemble model consensus to adaptively filter low-reliability virtual samples during training. By reducing RL's dependency on real state samples, this approach effectively generates optimal FL splitting strategies for resource-constrained IoT deployments.

![framework (1)](https://github.com/user-attachments/assets/837f973f-a3dd-4a09-873d-890ad42668ca)


The SplitRLC running on each client comprises three modules, namely: 1) Preprocessor; 2) Trained RL agent, and 3) Postprocessor. After completing an FL round (Round $t-1$), the preprocessor gathers device information, encompassing computational capabilities and network bandwidth, while also normalizing training time per iteration. The trained RL agent generates the offloading action after obtaining the observation (referred to as state). $\text{Section III-C}$ further discusses the training process of the RL agent. The post-processor uses the output of the trained RL model to assign offloading strategies to the devices in each group, ensuring they all follow the same action. This action determines which DNN model layer to split, providing a flexible and dynamic offloading method for FSL. Clearly, the trained RL agent is essential to this framework.



More information on the states, observations, rewards, actions, offloading strategy, and the SplitRLC modules are presented in the research article entitled, "Adaptive Offloading with Sample-Efficient Reinforcement learning for Federated Split Learning".

### Code Structure

The repository contains the source code of SplitRLC. The code is organised as:

1. Federated Learning training code using SplitRLC in `FL_training` folder.
2. Reinforcement learning training code for SplitRLC agent in `RL_training` folder.

The results are saved as pickle files in the `results` folder.

All configuration options are given in `config.py` , which contains the architecture, model, FL training hyperparameters and RL training hyperparameters.

Currently, CIFAR10 dataset and Convolutional Neural Network (CNN) models are supported. The code can be extended to support other datasets and models.

### Setting up the environment

The IoT edge server environment includes one server and five devices. The server is the same as used in Section V-B. The devices are: 1) One Raspberry Pi4 (Pi4) from Section V-B; 2) One RDK X3 development board with a 1.5GHz quad-core ARM Cortex A53 CPU (RDK); 3) Two Raspberry Pi3 Model B+ units with a 1.4GHz quad-core ARM Cortex-A53 CPU ; 4) A Jetson Xavier NX with an embedded GPU (Jetson). The RDK CPU frequency was set to 0.7GHz to create controlled stragglers. The Raspberry Pi and RDK development boards use the same operating system, Python 3.8, and PyTorch 1.4.0. The Jetson and the server also have the same versions of Python and PyTorch, with the CuDNN library installed on Jetson for GPU acceleration. All devices are connected to the server via Wi-Fi through a router, with an average bandwidth of 75Mb/s. The experiments were conducted in a real-world setting with five heterogeneous IoT devices, similar to the testbed used in the peer-reviewed FL study.

Then, modify the respective hostname and ip address in `config.py`. CLIENTS_CONFIG and CLIENTS_LIST in `config.py` are used for indexing and sorting.



```python
# Network configration
SERVER_ADDR= '192.168.110.35'
SERVER_PORT = 51000

K = 5# Number of devices
G = 3 # Number of groups

# Unique clients order
HOST2IP = {'splitrlc_node1':'192.168.110.30', 'splitrlc_node2':'192.168.110.31' , 'splitrlc_node3':'192.168.110.32', 'splitrlc_node4':'192.168.110.33','splitrlc_node5':'192.168.110.34' }
CLIENTS_CONFIG= {'192.168.110.30':0, '192.168.110.31':1, '192.168.110.32':2, '192.168.110.33':3,'192.168.110.34':4}
CLIENTS_LIST= [ '192.168.110.30', '192.168.110.31', '192.168.110.32', '192.168.110.33','192.168.110.34']
```

To test the code:

- Run FL training using SplitRLC: please follow instructions in `FL_training` folder.
- Run RL training for SplitRLC agent: please follow instructions in `RL_training` folder.

