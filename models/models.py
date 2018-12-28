
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO Implements other models of the paper


class SimpleDQNet(nn.Module):

    def __init__(self, stack_size, nb_actions):

        self.stack_size = stack_size
        self.nb_actions = nb_actions

        super(SimpleDQNet, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=self.stack_size,
            out_channels=32,
            kernel_size=8,
            stride=4
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2
        )

        self.fc1 = nn.Linear(
            in_features=64 * 5 * 4,   # TODO adapt to the stack size
            out_features=512
        )

        self.fc2 = nn.Linear(
            in_features=512,
            out_features=self.nb_actions
        )

    def forward(self, state):
        x = torch.Tensor(state)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
