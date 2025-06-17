import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np
import logging
from collections import deque
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.next_states = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.next_states[:]


class TransitionModel(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(TransitionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)

    def forward(self, latent_state, action):
        x = torch.cat([latent_state, action], dim=-1)
        return self.fc(x)


class BackwardTransitionModel(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(BackwardTransitionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

    def forward(self, latent_state, action):
        x = torch.cat([latent_state, action], dim=-1)
        return self.fc(x)


class Encoder(nn.Module):
    def __init__(self, state_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

    def forward(self, state):
        return self.fc(state)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std, latent_dim=32, num_ensemble=5):
        super(ActorCritic, self).__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Encoder network
        self.encoder = Encoder(state_dim, latent_dim)

        # Transition models
        self.transition_models = nn.ModuleList([
            TransitionModel(latent_dim, action_dim) for _ in range(num_ensemble)
        ])

        # Backward transition model
        self.backward_model = BackwardTransitionModel(latent_dim, action_dim)

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Sigmoid()
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Action variance
        self.init_action_std = action_std
        self.action_std = action_std
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def encode(self, state):
        return self.encoder(state)

    def predict_future(self, latent_state, actions, model_idx=None):
        """Predict future states using the transition model"""
        if model_idx is not None:
            # Use specific ensemble member
            model = self.transition_models[model_idx]
            pred_states = []
            current_state = latent_state
            for action in actions:
                current_state = model(current_state, action)
                pred_states.append(current_state)
            return torch.stack(pred_states)
        else:
            # Use all ensemble members
            ensemble_preds = []
            for model in self.transition_models:
                pred_states = []
                current_state = latent_state
                for action in actions:
                    current_state = model(current_state, action)
                    pred_states.append(current_state)
                ensemble_preds.append(torch.stack(pred_states))
            return torch.stack(ensemble_preds)  # [num_ensemble, pred_steps, latent_dim]

    def predict_past(self, latent_state, actions):
        """Predict past states using the backward transition model"""
        pred_states = []
        current_state = latent_state
        for action in reversed(actions):
            current_state = self.backward_model(current_state, action)
            pred_states.insert(0, current_state)
        return torch.stack(pred_states)

    def act(self, state, memory):
        latent_state = self.encode(state)
        action_mean = self.actor(latent_state)
        cov_mat = torch.diag(self.action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach(), action_mean.detach()

    def evaluate(self, state, action):
        latent_state = self.encode(state)
        action_mean = self.actor(latent_state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(latent_state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def std_decay(self, epoch):
        self.action_std = self.init_action_std * (0.9 ** epoch)
        self.action_var = torch.full((self.action_dim,), self.action_std * self.action_std).to(device)


class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip,
                 num_ensemble=5, pred_steps=5, num_virtual_samples=20, warmup_steps=500):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.pred_steps = pred_steps
        self.num_virtual_samples = num_virtual_samples
        self.warmup_steps = warmup_steps
        self.current_step = 0

        # Initialize policy
        self.policy = ActorCritic(state_dim, action_dim, action_std, num_ensemble=num_ensemble).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        # Old policy
        self.policy_old = ActorCritic(state_dim, action_dim, action_std, num_ensemble=num_ensemble).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Loss functions
        self.MseLoss = nn.MSELoss()

        # Replay buffer for real samples
        self.real_buffer = deque(maxlen=10000)

        # Hyperparameters
        self.gamma_pred = 1.0  # Prediction loss weight
        self.gamma_ccl_max = 1.0  # Max cycle consistency loss weight
        self.lambda_reliability = 1.0  # Reliability sensitivity parameter

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
        latent_state = self.policy.encode(state)
        action_mean = self.policy.actor(latent_state)
        return action_mean[0].cpu().data.numpy().flatten()

    def store_real_sample(self, state, action, next_state, reward, done):
        self.real_buffer.append((state, action, next_state, reward, done))

    def compute_prediction_loss(self, states, actions, next_states):
        """Compute the forward prediction loss for transition models"""
        latent_states = self.policy.encode(states)
        latent_next_states = self.policy.encode(next_states)

        # Predict next states using each transition model
        pred_loss = 0
        for model in self.policy.transition_models:
            pred_next = model(latent_states, actions)
            pred_loss += F.mse_loss(pred_next, latent_next_states.detach())

        # Also train backward model
        pred_prev = self.policy.backward_model(latent_next_states, actions)
        pred_loss += F.mse_loss(pred_prev, latent_states.detach())

        return pred_loss / (len(self.policy.transition_models) + 1)

    def compute_cycle_consistency_loss(self, states):
        """Compute cycle consistency loss with reliability filtering"""
        if len(states) < self.pred_steps + 1:
            return torch.tensor(0.0).to(device)

        # Select random states from buffer
        idx = random.randint(0, len(states) - self.pred_steps - 1)
        state_seq = states[idx:idx + self.pred_steps + 1]

        # Encode states
        latent_states = torch.stack([self.policy.encode(torch.FloatTensor(s).to(device)) for s in state_seq])

        # Generate random actions
        random_actions = torch.rand(self.num_virtual_samples, self.pred_steps, self.policy.action_dim).to(device)

        total_loss = 0
        total_reliability = 0

        for i in range(self.num_virtual_samples):
            actions = random_actions[i]

            # Forward prediction
            pred_future = []
            current_state = latent_states[0]
            for a in actions:
                current_state = self.policy.transition_models[0](current_state, a)  # Use first model for forward
                pred_future.append(current_state)
            pred_future = torch.stack(pred_future)

            # Backward prediction
            pred_past = []
            current_state = pred_future[-1]
            for a in reversed(actions):
                current_state = self.policy.backward_model(current_state, a)
                pred_past.insert(0, current_state)
            pred_past = torch.stack(pred_past)

            # Compute reliability score using ensemble
            ensemble_preds = []
            for model in self.policy.transition_models:
                current_state = latent_states[0]
                model_preds = []
                for a in actions:
                    current_state = model(current_state, a)
                    model_preds.append(current_state)
                ensemble_preds.append(torch.stack(model_preds))

            # Compute mean prediction and distances
            ensemble_preds = torch.stack(ensemble_preds)
            mean_pred = torch.mean(ensemble_preds, dim=0)
            distances = 1 - F.cosine_similarity(ensemble_preds, mean_pred.unsqueeze(0), dim=-1)
            avg_distance = torch.mean(distances)
            reliability = torch.exp(-self.lambda_reliability * avg_distance)

            # Cycle consistency loss
            cycle_loss = 1 - F.cosine_similarity(pred_past[0], latent_states[0], dim=-1)

            total_loss += reliability * cycle_loss
            total_reliability += reliability

        if total_reliability > 0:
            return total_loss / total_reliability
        return torch.tensor(0.0).to(device)

    def update(self, memory):
        # Store real samples in buffer
        for i in range(len(memory.states)):
            if i < len(memory.next_states):
                self.store_real_sample(
                    memory.states[i],
                    memory.actions[i],
                    memory.next_states[i],
                    memory.rewards[i],
                    memory.is_terminals[i]
                )

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

        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Update current step counter
            self.current_step += 1

            # Compute losses
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # RL loss
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            rl_loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # Prediction loss (only if we have enough real samples)
            pred_loss = torch.tensor(0.0).to(device)
            if len(self.real_buffer) > self.pred_steps + 1:
                # Sample a batch of real transitions
                samples = random.sample(self.real_buffer, min(32, len(self.real_buffer)))
                states = torch.FloatTensor(np.array([s[0] for s in samples])).to(device)
                actions = torch.FloatTensor(np.array([s[1] for s in samples])).to(device)
                next_states = torch.FloatTensor(np.array([s[2] for s in samples])).to(device)

                pred_loss = self.compute_prediction_loss(states, actions, next_states)

            # Cycle consistency loss
            ccl_loss = torch.tensor(0.0).to(device)
            if len(self.real_buffer) > self.pred_steps + 1:
                ccl_loss = self.compute_cycle_consistency_loss([s[0] for s in self.real_buffer])

            # Adjust cycle consistency weight based on warmup
            if self.current_step < self.warmup_steps:
                gamma_ccl = self.gamma_ccl_max * np.exp(-5 * (1 - self.current_step / self.warmup_steps) ** 2)
            else:
                gamma_ccl = self.gamma_ccl_max

            # Total loss
            total_loss = rl_loss.mean() + self.gamma_pred * pred_loss + gamma_ccl * ccl_loss

            # Take gradient step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
