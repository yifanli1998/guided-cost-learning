import torch.nn as nn
import torch


class CostNN(nn.Module):

    def __init__(
        self,
        state_dim,
        hidden_dim1=128,
        # hidden_dim2 = 128,
        out_features=1,
    ):
        super(CostNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim1),
            nn.ReLU(),
            # nn.Linear(hidden_dim1, hidden_dim2),
            # nn.ReLU(),
            nn.Linear(hidden_dim1, out_features),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

    def save_model(self, path):
        torch.save(self.net.state_dict(), path)

    def load_model(self, path):
        self.net.load_state_dict(torch.load(path))
