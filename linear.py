from torch import nn
from torch.nn import Linear


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = Linear(196608, 10)

