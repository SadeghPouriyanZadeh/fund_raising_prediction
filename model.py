from torch import nn

class IcoPredictor(nn.Module):
    def __init__(self, features, shallow_nodes) -> None:
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(features, shallow_nodes),
            nn.ReLU(),
            nn.Linear(shallow_nodes, 1),
            nn.ReLU(),

        )
    def forward(self, x):
        return self.regressor(x)