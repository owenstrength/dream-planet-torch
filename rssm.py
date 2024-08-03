import torch
import torch.nn as nn
import torch.nn.functional as F

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