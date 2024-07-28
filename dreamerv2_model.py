import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

class VisualEncoder(nn.Module):
    def __init__(self, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        if embedding_size == 1024:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(1024, embedding_size)

    def forward(self, observation):
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        hidden = self.fc(hidden)
        return hidden


class VisualDecoder(nn.Module):
    def __init__(self, state_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(state_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, state):
        hidden = self.fc1(state)
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        return observation


class WorldModel(nn.Module):
    def __init__(self, action_size, state_size=200, latent_size=30, hidden_size=200, embed_size=1024):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.encoder = VisualEncoder(embed_size)
        self.decoder = VisualDecoder(latent_size, embed_size)
        self.dynamics = RSSM(action_size, state_size, latent_size, hidden_size, embed_size)
        self.reward = RewardPredictor(latent_size, hidden_size)

    def imagine_ahead(self, post, horizon):
        flatten = lambda x: x.reshape(-1, *x.shape[2:])
        start = {k: flatten(v) for k, v in post.items()}
        def step(prev, _):
            state = self.dynamics.img_step(prev['deter'], prev['stoch'])
            return {**state, 'deter': state['deter'], 'stoch': state['stoch']}
        trajec = {k: [v] for k, v in start.items()}
        for _ in range(horizon):
            prev = {k: v[-1] for k, v in trajec.items()}
            next_state = step(prev, None)
            for k, v in next_state.items():
                trajec[k].append(v)
        trajec = {k: torch.stack(v, 0) for k, v in trajec.items()}
        return trajec

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

class RSSM(nn.Module):
    def __init__(self, action_size, state_size, latent_size, hidden_size, embed_size):
        super().__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        
        self.deter_cell = nn.GRUCell(embed_size + action_size, state_size)
        self.fc_prior = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2 * latent_size)
        )
        self.fc_posterior = nn.Sequential(
            nn.Linear(state_size + embed_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2 * latent_size)
        )

    def _sampleDist(self, mean, std):
        std = F.softplus(std) + 0.1  # Ensure std is positive
        return Independent(Normal(mean, std), 1)

    def observe(self, embed, action, deter=None):
        # Trim embed to match action sequence length
        embed = embed[:, :action.size(1)]
        
        batch_size, seq_len, _ = action.size()
        
        if deter is None:
            deter = torch.zeros(batch_size, self.state_size, device=embed.device)
        
        priors = []
        posteriors = []
        deters = []
        stochs = []

        for t in range(seq_len):
            deter = self.deter_cell(torch.cat([action[:, t], embed[:, t]], -1), deter)
            prior_mean, prior_std = torch.chunk(self.fc_prior(deter), 2, -1)
            posterior_mean, posterior_std = torch.chunk(self.fc_posterior(torch.cat([deter, embed[:, t]], -1)), 2, -1)
            
            prior = self._sampleDist(prior_mean, prior_std)
            posterior = self._sampleDist(posterior_mean, posterior_std)
            stoch = posterior.rsample()
            
            priors.append(prior)
            posteriors.append(posterior)
            deters.append(deter)
            stochs.append(stoch)

        priors = Independent(Normal(*[torch.stack([p.base_dist.loc for p in priors], 1),
                                      torch.stack([p.base_dist.scale for p in priors], 1)]), 2)
        posteriors = Independent(Normal(*[torch.stack([p.base_dist.loc for p in posteriors], 1),
                                          torch.stack([p.base_dist.scale for p in posteriors], 1)]), 2)
        deters = torch.stack(deters, 1)
        stochs = torch.stack(stochs, 1)

        return {'deter': deters, 'stoch': stochs, 'prior': priors, 'posterior': posteriors}

    def imagine(self, action, deter, stoch):
        next_deter = self.deter_cell(torch.cat([action, stoch], -1), deter)
        prior_mean, prior_std = torch.chunk(self.fc_prior(next_deter), 2, -1)
        prior = self._sampleDist(prior_mean, prior_std)
        next_stoch = prior.rsample()
        return {'deter': next_deter, 'stoch': next_stoch, 'prior': prior}

    def get_feat(self, state):
        return torch.cat([state['deter'], state['stoch']], -1)

class RewardPredictor(nn.Module):
    def __init__(self, latent_size, hidden_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, latent):
        return self.fc(latent).squeeze(-1)

class Actor(nn.Module):
    def __init__(self, state_size, latent_size, action_size, hidden_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size + latent_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2 * action_size)
        )
        self.action_size = action_size

    def forward(self, state, latent):
        x = torch.cat([state, latent], -1)
        mean, std = torch.chunk(self.fc(x), 2, -1)
        std = F.softplus(std) + 0.1
        return Independent(Normal(mean, std), 1)

class Critic(nn.Module):
    def __init__(self, state_size, latent_size, hidden_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size + latent_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, latent):
        x = torch.cat([state, latent], -1)
        return self.fc(x).squeeze(-1)
    
