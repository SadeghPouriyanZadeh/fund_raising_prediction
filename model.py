from torch import nn

class CancerDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(15, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x