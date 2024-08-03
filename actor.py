import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

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