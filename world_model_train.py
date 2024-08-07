import torch
import torch.nn.functional as F
import gym
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter


from world_model import WorldModel
from replay_buffer import ReplayBuffer

torch.autograd.set_detect_anomaly(True)


class DreamerV2:
    def __init__(self, state_dim, action_dim, rnn_hidden_dim, device='cpu', is_continous=True):
        self.device = device
        self.is_continous = is_continous

        self.world_model = WorldModel(state_dim, action_dim, rnn_hidden_dim).to(device)

        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(100000)

        # Hyperparameters
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.001
        self.lambda_value = 0.95
        self.rho = 0.5
        self.eta = 0.01
        self.imagine_horizon = 50
        self.entropy_coef = 0.01

        self.writer = SummaryWriter()

    def update_world_model(self, states, actions, rewards, next_states):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)

        hidden = torch.zeros(states.size(0), self.world_model.rssm.rnn_hidden_dim).to(self.device)
       
        state_preds, reward_preds, _ = self.world_model(states, actions, hidden)
       
        state_loss = F.mse_loss(state_preds, next_states)
        reward_loss = F.mse_loss(reward_preds, rewards)
       
        total_loss = state_loss + reward_loss

        self.world_model_optimizer.zero_grad()
        total_loss.backward()
        self.world_model_optimizer.step()

        return total_loss.item(), state_loss.item(), reward_loss.item()


    def train(self, env, num_episodes, max_steps_per_episode=200):
        for episode in range(num_episodes):
            state = env.reset()[0]
            total_reward = 0
            episode_steps = 0

            for step in range(max_steps_per_episode):
                action = env.action_space.sample()
                if self.is_continous:
                    next_state, reward, done, truncated, _ = env.step(action[0])
                else:
                    next_state, reward, done, truncated, _ = env.step(action)
                    idx = action
                    action = np.asarray([0] * env.action_space.n)
                    action[idx] = 1

                if done or truncated:
                    break

                if reward > 0:
                    reward = -reward
                self.replay_buffer.push(state, action, reward, next_state, done)

                if len(self.replay_buffer) > self.batch_size:
                    batch = self.replay_buffer.sample(self.batch_size)
                    batch_states, batch_actions, batch_rewards, batch_next_states, _ = zip(*batch)

                    world_model_loss, state_loss, reward_loss = self.update_world_model(
                        batch_states, batch_actions, batch_rewards, batch_next_states)

                    self.writer.add_scalar('Loss/World_Model', world_model_loss, episode * max_steps_per_episode + step)
                    self.writer.add_scalar('Loss/State_Prediction', state_loss, episode * max_steps_per_episode + step)
                    self.writer.add_scalar('Loss/Reward_Prediction', reward_loss, episode * max_steps_per_episode + step)

                state = next_state
                total_reward += reward
                episode_steps += 1

                if done or truncated:
                    break

            self.writer.add_scalar('Reward/Episode', total_reward, episode)
            self.writer.add_scalar('Steps/Episode', episode_steps, episode)

            print(f"Episode {episode + 1}/{num_episodes}, Steps: {episode_steps}, Total Reward: {total_reward:.2f}")


        self.writer.close()
        torch.save(self.world_model.state_dict(), './models/world_model.pth')
        print('Model saved successfully')

# Usage
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
is_continous = False if isinstance(env.action_space, gym.spaces.Discrete) else True
action_dim = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]
rnn_hidden_dim = 256

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")
print(f"Using device: {device}")
dreamer = DreamerV2(state_dim, action_dim, rnn_hidden_dim, device, is_continous=is_continous)
dreamer.train(env, num_episodes=500, max_steps_per_episode=200)