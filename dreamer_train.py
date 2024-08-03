import torch
import torch.nn.functional as F
import gym
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter


from world_model import WorldModel
from actor import Actor
from critic import Critic
from replay_buffer import ReplayBuffer

torch.autograd.set_detect_anomaly(True)


class DreamerV2:
    def __init__(self, state_dim, action_dim, rnn_hidden_dim, device='cpu', is_continous=True):
        self.device = device
        self.is_continous = is_continous

        self.world_model = WorldModel(state_dim, action_dim, rnn_hidden_dim).to(device)
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.target_critic = Critic(state_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=1e-3)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-5)

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

    def compute_lambda_returns(self, rewards, states):
        # Use target network for value estimation
        with torch.no_grad():
            values = self.target_critic(states).squeeze(-1).to(self.device)
        
        # Normalize rewards
        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        lambda_returns = torch.zeros_like(rewards).to(self.device)
        last_lambda_return = values[-1]

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                lambda_returns[t] = rewards[t] + self.gamma * values[t]
            else:
                lambda_returns[t] = rewards[t] + self.gamma * (
                    (1 - self.lambda_value) * values[t + 1] +
                    self.lambda_value * last_lambda_return)
            last_lambda_return = lambda_returns[t]

        return lambda_returns.view(-1)

    def update_critic(self, imagined_states, imagined_rewards):
        values = self.critic(imagined_states[:-1]).squeeze(-1).to(self.device)
        
        lambda_returns = self.compute_lambda_returns(imagined_rewards, imagined_states)
        
        critic_loss = 0.5 * F.mse_loss(values[:-1], lambda_returns[:-1].detach())

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # Soft update target network
        self.soft_update_target_network()

        return critic_loss.item()
    
    def update_actor(self, imagined_states, imagined_rewards, log_probs):
        # Compute values and lambda returns
        with torch.no_grad():
            values = self.critic(imagined_states[:-1]).squeeze(-1).to(self.device)
            lambda_returns = self.compute_lambda_returns(imagined_rewards, imagined_states[:-1])

        # Compute advantages
        advantages = lambda_returns - values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.clamp(advantages, -1, 1)

        # Reinforce term
        reinforce_loss = -self.rho * (log_probs * advantages.detach()).mean()

        # Dynamics backprop term
        dynamics_loss = -(1 - self.rho) * lambda_returns.mean()

        # Entropy regularizer
        entropy = -log_probs.mean()
        entropy_loss = -self.eta * entropy

        # Combined actor loss
        actor_loss = reinforce_loss + dynamics_loss + entropy_loss

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        return actor_loss.item(), entropy.item()

    def soft_update_target_network(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train(self, env, num_episodes, max_steps_per_episode=200):
        for episode in range(num_episodes):
            state = env.reset()[0]
            hidden = torch.zeros(1, self.world_model.rssm.rnn_hidden_dim).to(self.device)
            total_reward = 0
            episode_steps = 0

            for step in range(max_steps_per_episode):
                action, _ = self.actor.sample(torch.FloatTensor(state).unsqueeze(0).to(self.device))
                action = action.squeeze(0).cpu().detach().numpy()
                if self.is_continous:
                    next_state, reward, done, truncated, _ = env.step(action)
                else:
                    next_state, reward, done, truncated, _ = env.step(np.argmax(action))

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
                   
                    imagined_states, imagined_rewards, actions, log_probs = self.imagine_trajectories(state, hidden)
                    imagined_states = imagined_states.detach()
                    imagined_rewards = imagined_rewards.detach()
                    actor_loss, entropy = self.update_actor(imagined_states, imagined_rewards, log_probs)
                    critic_loss = self.update_critic(imagined_states, imagined_rewards)

                    self.writer.add_scalar('Loss/World_Model', world_model_loss, episode * max_steps_per_episode + step)
                    self.writer.add_scalar('Loss/State_Prediction', state_loss, episode * max_steps_per_episode + step)
                    self.writer.add_scalar('Loss/Reward_Prediction', reward_loss, episode * max_steps_per_episode + step)
                    self.writer.add_scalar('Loss/Actor', actor_loss, episode * max_steps_per_episode + step)
                    self.writer.add_scalar('Loss/Critic', critic_loss, episode * max_steps_per_episode + step)
                    self.writer.add_scalar('Entropy', entropy, episode * max_steps_per_episode + step)

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
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
is_continous = False if isinstance(env.action_space, gym.spaces.Discrete) else True
action_dim = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]
rnn_hidden_dim = 256

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")
print(f"Using device: {device}")
dreamer = DreamerV2(state_dim, action_dim, rnn_hidden_dim, device, is_continous=is_continous)
dreamer.train(env, num_episodes=1000, max_steps_per_episode=200)