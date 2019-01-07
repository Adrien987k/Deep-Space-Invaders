
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
            in_features=64 * 5 * 4,   # TODO adapt to the stack size, also on the ameliorated versions ...
            out_features=512
        )

        self.fc2 = nn.Linear(
            in_features=512,
            out_features=self.nb_actions
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

#Duelling double DQ-net
class DDDQNet(nn.Module):

    def __init__(self, stack_size, nb_actions):

        self.stack_size = stack_size
        self.nb_actions = nb_actions

        super(DDDQNet, self).__init__()

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

        # We now have two streams.
        # The one that calculate V(s)

        self.value_fc = nn.Linear(
            in_features=64 * 5 * 4,   # TODO adapt to the stack size, also on the ameliorated versions ...
            out_features=512
        )

        self.value = nn.Linear(
            in_features = 512,
            out_features = 1
        )

        # The one doing A(s, a)

        self.advantage_fc = nn.Linear(
            in_features = 64 * 5 * 4,   # TODO adapt to the stack size
            out_features = 512
        )

        self.advantage = nn.Linear(
            in_features = 512,   # TODO adapt to the stack size
            out_features = self.nb_actions
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        val = F.relu(self.value_fc(x))
        val = self.value(val)

        adv = F.relu(self.advantage_fc(x))
        adv = self.advantage(adv)

        return val + adv - torch.mean(adv, dim = 1, keepdim = True)
