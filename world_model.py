import torch.nn as nn
from rssm import RSSM

class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, rnn_hidden_dim):
        super().__init__()
        self.rssm = RSSM(state_dim, action_dim, rnn_hidden_dim)
        self.reward_predictor = nn.Linear(state_dim, 1)

    def forward(self, state, action, hidden):
        state_pred, hidden = self.rssm(state, action, hidden)
        reward_pred = self.reward_predictor(state_pred)
        return state_pred, reward_pred, hidden