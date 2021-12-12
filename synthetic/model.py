"""System module."""
from torch import nn


class Shallow(nn.Module):
    def __init__(self, nhidden=8):
        # build two-layer fully connected network
        super(Shallow, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, nhidden),
            nn.ReLU(inplace=True),
            nn.Linear(nhidden, nhidden),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Linear(nhidden, 1)
        self.ssh = nn.Linear(nhidden, 1)

    def forward(self, x):
        # compute main and ssl task output and encoder output
        h = self.encoder(x)
        return self.cls(h), self.ssh(h), h
