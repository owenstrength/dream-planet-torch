import pdb
import torch
from tqdm import trange
from functools import partial
from collections import defaultdict


from torch.distributions import Normal, kl
from torch.distributions.kl import kl_divergence

from ac_models import Actor, Critic
from dreamerv2_policy import DreamerPolicy
from utils import *
from memory import *
from rssm_model import *
from rssm_policy import *
from rollout_generator import RolloutGenerator

def train(memory, rssm, actor, critic, optimizer, optimizer_actor, optimizer_critic, device, N=32, H=50, beta=1.0, grads=False):
    free_nats = torch.ones(1, device=device)*3.0
    batch = memory.sample(N, H, time_first=True)
    x, u, r, t = [torch.tensor(x).float().to(device) for x in batch]
    preprocess_img(x, depth=5)
    e_t = bottle(rssm.encoder, x)
    h_t, s_t = rssm.get_init_state(e_t[0])
    kl_loss, rc_loss, re_loss = 0, 0, 0
    states, priors, posteriors, posterior_samples, actions = [], [], [], [], []
    
    # Forward pass through RSSM
    for i, a_t in enumerate(torch.unbind(u, dim=0)):
        h_t = rssm.deterministic_state_fwd(h_t, s_t, a_t)
        states.append(h_t)
        priors.append(rssm.state_prior(h_t))
        posteriors.append(rssm.state_posterior(h_t, e_t[i + 1]))
        posterior_samples.append(Normal(*posteriors[-1]).rsample())
        actions.append(actor(h_t.detach()))  # Detach h_t to prevent gradients flowing back to RSSM
        s_t = posterior_samples[-1]
    
    prior_dist = Normal(*map(torch.stack, zip(*priors)))
    posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
    states, posterior_samples = map(torch.stack, (states, posterior_samples))
    
    # Compute RSSM losses
    rec_loss = F.mse_loss(
        bottle(rssm.decoder, states, posterior_samples), x[1:],
        reduction='none'
    ).sum((2, 3, 4)).mean()
    
    kld_loss = torch.max(
        kl_divergence(posterior_dist, prior_dist).sum(-1),
        free_nats
    ).mean()
    
    rew_loss = F.mse_loss(
        bottle(rssm.pred_reward, states, posterior_samples), r
    )
    
    # Update RSSM
    optimizer.zero_grad()
    rssm_loss = beta * kld_loss + rec_loss + rew_loss
    rssm_loss.backward()
    nn.utils.clip_grad_norm_(rssm.parameters(), 1000., norm_type=2)
    optimizer.step()
    
    # Update Actor-Critic
    states_detached = states.detach()
    actions_stacked = torch.stack(actions)
    update_actor_critic(actor, critic, optimizer_actor, optimizer_critic, states_detached, actions_stacked, r, states_detached[1:], t)
    
    metrics = {
        'losses': {
            'kl': kld_loss.item(),
            'reconstruction': rec_loss.item(),
            'reward_pred': rew_loss.item()
        },
    }
    if grads:
        metrics['grad_norms'] = {
            k: 0 if v.grad is None else v.grad.norm().item()
            for k, v in rssm.named_parameters()
        }
    return metrics


def main():
    env = TorchImageEnvWrapper('Pendulum-v1', bit_depth=5)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    rssm_model = RecurrentStateSpaceModel(
        action_size=env.action_size,
    ).to(device)
    
    # Define the Actor and Critic networks
    actor = Actor(
        state_dim=rssm_model.state_size,
        action_dim=env.action_size,
    ).to(device)
    
    critic = Critic(
        state_dim=rssm_model.state_size,
        action_dim=env.action_size,
    ).to(device)
    
    optimizer_rssm = torch.optim.Adam(rssm_model.parameters(), lr=1e-3, eps=1e-4)
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
    
    policy = DreamerPolicy(
        actor,
        rssm_model,
        planning_horizon=20,
        num_candidates=1000,
        num_iterations=10,
        top_candidates=100,
        device=device
    )
    rollout_gen = RolloutGenerator(
        env,
        device,
        policy=policy,
        episode_gen=lambda : Episode(partial(postprocess_img, depth=5)),
        max_episode_steps=100,
    )
    mem = Memory(100)
    mem.append(rollout_gen.rollout_n(1, random_policy=True))
    res_dir = 'results_3/'
    summary = TensorBoardMetrics(f'{res_dir}/')
    
    for i in trange(100, desc='Epoch', leave=False):
        metrics = {}
        for _ in trange(150, desc='Iter ', leave=False):
            train_metrics = train(mem, rssm_model.train(), actor, critic, optimizer_rssm, optimizer_actor, optimizer_critic, device)
            for k, v in flatten_dict(train_metrics).items():
                if k not in metrics.keys():
                    metrics[k] = []
                metrics[k].append(v)
                metrics[f'{k}_mean'] = np.array(v).mean()
        
        summary.update(metrics)
        mem.append(rollout_gen.rollout_once(explore=True))
        eval_episode, eval_frames, eval_metrics = rollout_gen.rollout_eval()
        mem.append(eval_episode)
        save_video(eval_frames, res_dir, f'vid_{i+1}')
        summary.update(eval_metrics)

        if (i + 1) % 25 == 0:
            torch.save(rssm_model.state_dict(), f'{res_dir}/ckpt_{i+1}.pth')
            torch.save(actor.state_dict(), f'{res_dir}/actor_{i+1}.pth')
            torch.save(critic.state_dict(), f'{res_dir}/critic_{i+1}.pth')

    pdb.set_trace()

if __name__ == '__main__':
    main()