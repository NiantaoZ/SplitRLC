import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Sigmoid()
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # model for forward and inverse dynamics
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.init_action_std = action_std
        self.action_std = action_std
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)
        logger.info(' Current action mean: ' + str(action_mean))

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        logger.info('Current action sample: ' + str(action))
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach(), action_mean.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def std_decay(self, epoch):
        self.action_std = self.init_action_std * (0.9 ** epoch)
        self.action_var = torch.full((self.action_dim,), self.action_std * self.action_std).to(device)


    def compute_forward_prediction_loss(self, states, actions):
        # Using the forward model to predict next states
        state_action = torch.cat((states, actions), dim=1)
        predicted_next_states = self.policy.forward_model(state_action)

        # Compute forward prediction loss (squared difference)
        forward_pred_loss = self.MseLoss(predicted_next_states, states)
        return forward_pred_loss

    def compute_backward_prediction_loss(self, states, actions):
        # Using inverse model to predict actions from state pairs
        state_pairs = torch.cat((states, states), dim=1)
        predicted_actions = self.policy.inverse_model(state_pairs)

        # Compute backward prediction loss (squared difference)
        backward_pred_loss = self.MseLoss(predicted_actions, actions)
        return backward_pred_loss

    def compute_real_cycle_loss(self, states, actions):
        # Real cycle involves predicting the next state and returning to the initial state
        state_action = torch.cat((states, actions), dim=1)
        predicted_next_states = self.policy.forward_model(state_action)
        predicted_initial_states = self.policy.inverse_model(
            torch.cat((predicted_next_states, states), dim=1)
        )

        # Compute real cycle loss (squared difference)
        real_cycle_loss = self.MseLoss(predicted_initial_states, states)
        return real_cycle_loss

    def compute_virtual_cycle_loss(self, states, actions):
        # Virtual cycle involves using predicted states and actions
        state_action = torch.cat((states, actions), dim=1)
        predicted_next_states = self.policy.forward_model(state_action)
        virtual_actions = self.policy.inverse_model(
            torch.cat((states, predicted_next_states), dim=1)
        )
        predicted_virtual_states = self.policy.forward_model(
            torch.cat((predicted_next_states, virtual_actions), dim=1)
        )

        # Compute virtual cycle loss (squared difference)
        virtual_cycle_loss = self.MseLoss(predicted_virtual_states, states)
        return virtual_cycle_loss


class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def explore_decay(self, epoch):
        self.policy.std_decay(epoch)
        self.policy_old.std_decay(epoch)

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        actions = self.policy_old.act(state, memory)
        stds = self.policy_old.action_var
        return actions[0].cpu().data.numpy().flatten(), actions[
            1].cpu().data.numpy().flatten(), stds.cpu().data.numpy().flatten()

    def exploit(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action_mean = self.policy.actor(state)
        return action_mean[0].cpu().data.numpy().flatten()

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values:
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            surrogate_loss = -torch.min(surr1, surr2)

            # Additional loss components
            forward_pred_loss = self.compute_forward_prediction_loss(old_states, old_actions)
            backward_pred_loss = self.compute_backward_prediction_loss(old_states, old_actions)
            real_cycle_loss = self.compute_real_cycle_loss(old_states, old_actions)
            virtual_cycle_loss = self.compute_virtual_cycle_loss(old_states, old_actions)

            # Combine all losses
            model_rl_loss = forward_pred_loss + backward_pred_loss + real_cycle_loss + virtual_cycle_loss
            total_loss = surrogate_loss.mean() + 0.5 * self.MseLoss(state_values,
                                                                    rewards) - 0.01 * dist_entropy + model_rl_loss

            # take gradient step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

