from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.actions_size = action_size
        self.hiddenOne = nn.Linear(64)
        self.hiddenTwo = nn.Linear(64)
        self.hiddenThree = nn.Linear(self.actions_size)


    def forward(self, input):
        x = input.view(-1, self.state_size)
        x = self.hiddenOne(x)
        x = F.relu(x)
        x = self.hiddenTwo(x)
        x = F.relu(x)
        x = self.hiddenOne(x)
        x = F.linear(x)
        return x.view(-1, x.size(1))
