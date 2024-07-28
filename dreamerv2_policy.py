import torch
import torch.nn.functional as F
from torch.distributions import Normal

class DreamerPolicy:
    def __init__(self,
                 actor,  # Use the actor network
                 rssm,
                 planning_horizon,
                 num_candidates,
                 num_iterations,
                 top_candidates,
                 device
                 ):
        super().__init__()
        self.actor = actor
        self.rssm = rssm
        self.N = num_candidates
        self.K = top_candidates
        self.T = num_iterations
        self.H = planning_horizon
        self.d = self.rssm.action_size
        self.device = device
        self.state_size = self.rssm.state_size
        self.latent_size = self.rssm.latent_size

    def reset(self):
        self.h = torch.zeros(1, self.state_size).to(self.device)
        self.s = torch.zeros(1, self.latent_size).to(self.device)
        self.a = torch.zeros(1, self.d).to(self.device)

    def _poll(self, obs):
        self.mu = torch.zeros(self.H, self.d).to(self.device)
        self.stddev = torch.ones(self.H, self.d).to(self.device)
        assert len(obs.shape) == 3, 'obs should be [CHW]'
        print("self.a: ", self.a.shape)
        print("self.h: ", self.h.shape)
        print("self.s: ", self.s.shape)
        self.h, self.s = self.rssm.get_init_state(
            self.rssm.encoder(obs[None]),
            self.h, self.s, self.a
        )
        for _ in range(self.T):
            rwds = torch.zeros(self.N).to(self.device)
            # Generate actions using the actor network
            actions = self.actor(self.h).repeat(self.N, 1)
            h_t = self.h.clone().expand(self.N, -1)
            s_t = self.s.clone().expand(self.N, -1)
            for a_t in torch.unbind(actions.unsqueeze(1), dim=1):
                print("a_t: ", a_t.shape)
                print("h_t: ", h_t.shape)
                print("s_t: ", s_t.shape)
                h_t = self.rssm.deterministic_state_fwd(h_t, s_t, a_t)
                s_t = self.rssm.state_prior(h_t, sample=True)
                rwds += self.rssm.pred_reward(h_t, s_t)
            _, k = torch.topk(rwds, self.K, dim=0, largest=True, sorted=False)
            self.mu = actions[k].mean(dim=1)
            self.stddev = actions[k].std(dim=0, unbiased=False)
        self.a = self.mu[0].unsqueeze(-1)
        print("self.a: MUL", self.a.shape)

    def poll(self, observation, explore=False):
        with torch.no_grad():
            self._poll(observation)
            if explore:
                self.a += torch.randn_like(self.a) * 0.3
            return self.a