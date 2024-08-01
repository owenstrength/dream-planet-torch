import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import gym
import numpy as np
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)

class RSSM(nn.Module):
    def __init__(self, state_dim, action_dim, rnn_hidden_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        self.encoder = nn.Linear(state_dim + action_dim, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.decoder = nn.Linear(rnn_hidden_dim, state_dim)

    def forward(self, state, action, hidden):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.encoder(x))
        hidden = self.rnn(x, hidden)
        state_pred = self.decoder(hidden)
        return state_pred, hidden

class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, rnn_hidden_dim):
        super().__init__()
        self.rssm = RSSM(state_dim, action_dim, rnn_hidden_dim)
        self.reward_predictor = nn.Linear(state_dim, 1)

    def forward(self, state, action, hidden):
        state_pred, hidden = self.rssm(state, action, hidden)
        reward_pred = self.reward_predictor(state_pred)
        return state_pred, reward_pred, hidden

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_mean = nn.Linear(64, action_dim)
        self.fc_logstd = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        logstd = self.fc_logstd(x)
        logstd = torch.clamp(logstd, -20, 2)
        return mean, logstd

    def sample(self, state):
        mean, logstd = self(state)
        std = torch.exp(logstd) + 1e-6
        normal = td.Normal(mean, std)
        action = normal.rsample()
        log_prob = normal.log_prob(action).sum(axis=-1)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DreamerV2:
    def __init__(self, state_dim, action_dim, rnn_hidden_dim, device='cpu'):
        self.device = device
        self.world_model = WorldModel(state_dim, action_dim, rnn_hidden_dim).to(device)
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.target_critic = Critic(state_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=1e-3)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-6)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-6)

        self.replay_buffer = ReplayBuffer(100000)
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005
        self.imagine_horizon = 10
        self.lambda_gae = 0.95

        self.writer = SummaryWriter()

    def update_world_model(self, states, actions, rewards, next_states):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        hidden = torch.zeros(states.size(0), self.world_model.rssm.rnn_hidden_dim).to(self.device)
       
        state_preds, reward_preds, _ = self.world_model(states, actions, hidden)
       
        state_loss = F.mse_loss(state_preds, next_states)
        reward_loss = F.mse_loss(reward_preds, rewards)
       
        total_loss = state_loss + reward_loss

        self.world_model_optimizer.zero_grad()
        total_loss.backward()
        self.world_model_optimizer.step()

        return total_loss.item(), state_loss.item(), reward_loss.item()

    def imagine_trajectories(self, initial_state, initial_hidden):
        state = torch.FloatTensor(initial_state).unsqueeze(0).to(self.device)
        hidden = initial_hidden.to(self.device)
        imagined_states = [state]
        imagined_rewards = []
        actions = []
        log_probs = []

        for _ in range(self.imagine_horizon):
            action, log_prob = self.actor.sample(state)
            state_pred, reward_pred, hidden = self.world_model(state, action, hidden)
            imagined_states.append(state_pred)
            imagined_rewards.append(reward_pred)
            actions.append(action)
            log_probs.append(log_prob)
            state = state_pred

        return (torch.cat(imagined_states, dim=0),
                torch.cat(imagined_rewards, dim=0),
                torch.cat(actions, dim=0),
                torch.cat(log_probs, dim=0))

    def compute_lambda_returns(self, rewards, values):
        lambda_returns = []
        last_lambda_return = values[-1]

        for t in reversed(range(len(rewards))):
            bootstrap = (values[t + 1] if t + 1 < len(values) else values[-1])
            lambda_returns.append(rewards[t] + self.gamma * (
                self.lambda_gae * last_lambda_return +
                (1 - self.lambda_gae) * bootstrap))
            last_lambda_return = lambda_returns[-1]

        lambda_returns = lambda_returns[::-1]
        return torch.cat(lambda_returns, dim=0)

    def update_critic(self, imagined_states, imagined_rewards):
        # Compute values and lambda returns
        values = self.critic(imagined_states[:-1]).squeeze(-1)
        lambda_returns = self.compute_lambda_returns(imagined_rewards, values)

        # Compute critic loss
        critic_loss = F.mse_loss(values, lambda_returns.detach())

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        return critic_loss.item()
    
    def update_actor(self, imagined_states, imagined_rewards, log_probs):
        values = self.critic(imagined_states[:-1]).squeeze(-1)
        lambda_returns = self.compute_lambda_returns(imagined_rewards, values)
        
        advantages = lambda_returns - values.detach()
        
        
        # Check for NaN or Inf values
        if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
            print("Warning: NaN or Inf detected in log_probs")
            log_probs = torch.nan_to_num(log_probs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            print("Warning: NaN or Inf detected in advantages")
            advantages = torch.nan_to_num(advantages, nan=0.0, posinf=1e6, neginf=-1e6)
        
        actor_loss = -(log_probs * advantages).mean()
        

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)   
        self.actor_optimizer.step()

        return actor_loss.item()

    def train(self, env, num_episodes, max_steps_per_episode=200):
        for episode in range(num_episodes):
            state = env.reset()[0]
            hidden = torch.zeros(1, self.world_model.rssm.rnn_hidden_dim).to(self.device)
            total_reward = 0
            episode_steps = 0

            for step in range(max_steps_per_episode):
                action, _ = self.actor.sample(torch.FloatTensor(state).unsqueeze(0).to(self.device))
                action = action.squeeze(0).cpu().detach().numpy()
                next_state, reward, done, truncated, _ = env.step(action)

                if done or truncated:
                    break

                self.replay_buffer.push(state, action, reward, next_state, done)

                if len(self.replay_buffer) > self.batch_size:
                    batch = self.replay_buffer.sample(self.batch_size)
                    batch_states, batch_actions, batch_rewards, batch_next_states, _ = zip(*batch)

                    world_model_loss, state_loss, reward_loss = self.update_world_model(
                        batch_states, batch_actions, batch_rewards, batch_next_states)
                   
                    imagined_states, imagined_rewards, actions, log_probs = self.imagine_trajectories(state, hidden)
                    imagined_states = imagined_states.detach()
                    imagined_rewards = imagined_rewards.detach()
                    actor_loss = self.update_actor(imagined_states, imagined_rewards, log_probs)
                    critic_loss = self.update_critic(imagined_states, imagined_rewards)

                    self.writer.add_scalar('Loss/World_Model', world_model_loss, episode * max_steps_per_episode + step)
                    self.writer.add_scalar('Loss/State_Prediction', state_loss, episode * max_steps_per_episode + step)
                    self.writer.add_scalar('Loss/Reward_Prediction', reward_loss, episode * max_steps_per_episode + step)
                    self.writer.add_scalar('Loss/Actor', actor_loss, episode * max_steps_per_episode + step)
                    self.writer.add_scalar('Loss/Critic', critic_loss, episode * max_steps_per_episode + step)

                state = next_state
                total_reward += reward
                episode_steps += 1

                if done or truncated:
                    break

            self.writer.add_scalar('Reward/Episode', total_reward, episode)
            self.writer.add_scalar('Steps/Episode', episode_steps, episode)

            print(f"Episode {episode + 1}/{num_episodes}, Steps: {episode_steps}, Total Reward: {total_reward:.2f}")

        self.writer.close()

# Usage
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
rnn_hidden_dim = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dreamer = DreamerV2(state_dim, action_dim, rnn_hidden_dim, device)
dreamer.train(env, num_episodes=1000, max_steps_per_episode=200)