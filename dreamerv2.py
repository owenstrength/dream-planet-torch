import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from tqdm import trange
from functools import partial
from collections import defaultdict

from utils import *
from memory import *
from rollout_generator import RolloutGenerator

# Import the new classes
from dreamerv2_model import WorldModel, Actor, Critic
from dreamerv2_policy import DreamerV2Policy

def train(memory, world_model, actor, critic, policy, optimizer_model, optimizer_actor, optimizer_critic, device, batch_size=50, seq_length=50):
    batch = memory.sample(batch_size, seq_length)
    obs, actions, rewards, dones = [torch.tensor(x).float().to(device) for x in batch]
    
    # Use bottle function for processing observations
    embed = bottle(world_model.encoder, obs)
    
    # World model update
    model_state = world_model.dynamics.observe(embed[:, :-1], actions)
    feat = world_model.dynamics.get_feat(model_state)
    latent = model_state['stoch']
    
    # Reshape latent to maintain sequence dimension
    batch_dim, seq_dim, latent_dim = latent.shape
    latent_reshaped = latent.reshape(batch_dim * seq_dim, latent_dim)
    
    image_pred = world_model.decoder(latent_reshaped)
    image_pred = image_pred.view(batch_dim, seq_dim, *image_pred.shape[1:])
    
    reward_pred = world_model.reward(latent_reshaped)
    reward_pred = reward_pred.view(batch_dim, seq_dim)
    
    # Adjust the predictions to match the available ground truth
    rec_loss = F.mse_loss(image_pred[:, :-1], obs[:, 1:-1])
    reward_loss = F.mse_loss(reward_pred[:, :-1], rewards[:, 1:])
    
    kl_loss = torch.mean(torch.distributions.kl.kl_divergence(model_state['posterior'], model_state['prior']))
    model_loss = rec_loss + reward_loss + 0.1 * kl_loss
    
    optimizer_model.zero_grad()
    model_loss.backward()
    optimizer_model.step()
    
    # Actor and Critic update
    # Actor and Critic update
    with torch.no_grad():
        imag_traj, imag_reward, imag_value = policy.imagine_trajectory()
    
    actor_loss = policy.compute_actor_loss(imag_traj, imag_reward, imag_value)
    critic_loss = policy.compute_critic_loss(imag_traj, imag_reward, imag_value)
    
    optimizer_actor.zero_grad()
    actor_loss.backward()
    optimizer_actor.step()
    
    optimizer_critic.zero_grad()
    critic_loss.backward()
    optimizer_critic.step()
    
    return {
        'model_loss': model_loss.item(),
        'rec_loss': rec_loss.item(),
        'reward_loss': reward_loss.item(),
        'kl_loss': kl_loss.item(),
        'actor_loss': actor_loss.item(),
        'critic_loss': critic_loss.item()
    }

def main():
    env = TorchImageEnvWrapper('Pendulum-v1', bit_depth=5)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the new models
    world_model = WorldModel(env.action_size).to(device)
    actor = Actor(world_model.state_size, world_model.latent_size, env.action_size, hidden_size=200).to(device)
    critic = Critic(world_model.state_size, world_model.latent_size, hidden_size=200).to(device)
    
    # Initialize the DreamerV2Policy
    policy = DreamerV2Policy(world_model, actor, critic, device=device)
    
    # Initialize optimizers
    optimizer_model = optim.Adam(world_model.parameters(), lr=1e-3, eps=1e-4)
    optimizer_actor = optim.Adam(actor.parameters(), lr=8e-5, eps=1e-5)
    optimizer_critic = optim.Adam(critic.parameters(), lr=8e-5, eps=1e-5)
    
    rollout_gen = RolloutGenerator(
        env,
        device,
        policy=policy,
        episode_gen=lambda: Episode(partial(postprocess_img, depth=5)),
        max_episode_steps=1000,
    )
    
    mem = Memory(100)
    mem.append(rollout_gen.rollout_n(1, random_policy=True)) 
    
    res_dir = 'results_2/'
    summary = TensorBoardMetrics(f'{res_dir}/')
    
    for i in trange(100, desc='Epoch', leave=False): 
        metrics = defaultdict(list)
        for _ in trange(100, desc='Iter ', leave=False):
            train_metrics = train(mem, world_model, actor, critic, policy, optimizer_model, optimizer_actor, optimizer_critic, device)
            for k, v in train_metrics.items():
                metrics[k].append(v)
        
        for k, v in metrics.items():
            metrics[f'{k}_mean'] = np.mean(v)
        
        summary.update(metrics)
        
        # Collect more data
        mem.append(rollout_gen.rollout_once(explore=True))
        
        # Evaluate
        if (i + 1) % 10 == 0:  # Evaluate every 10 epochs
            eval_episode, eval_frames, eval_metrics = rollout_gen.rollout_eval()
            mem.append(eval_episode)
            save_video(eval_frames, res_dir, f'vid_{i+1}')
            summary.update(eval_metrics)
        
        # Save checkpoint
        if (i + 1) % 100 == 0:  # Save every 100 epochs
            torch.save({
                'world_model': world_model.state_dict(),
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'optimizer_model': optimizer_model.state_dict(),
                'optimizer_actor': optimizer_actor.state_dict(),
                'optimizer_critic': optimizer_critic.state_dict(),
            }, f'{res_dir}/ckpt_{i+1}.pth')

if __name__ == '__main__':
    main()