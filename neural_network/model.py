from torch import nn


class LinearReLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.seq(x)


class IcoPredictor(nn.Module):
    def __init__(self, features, hidden_layers, layer_units) -> None:
        super().__init__()
        self.regressor = nn.Sequential(
            LinearReLU(features, layer_units),
            *[LinearReLU(layer_units, layer_units) for _ in range(hidden_layers)],
            LinearReLU(layer_units, 1)
        )

    def forward(self, x):
        return self.regressor(x)
