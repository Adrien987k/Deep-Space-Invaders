
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO Implements other models of the paper


class SimpleDQNet(nn.Module):

    def __init__(self, nb_state, nb_action):

        self.nb_state = nb_state
        self.nb_action = nb_action

        super(SimpleDQNet, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=self.nb_action,
            out_channels=20 * 20,
            kernel_size=8,
            stride=4
        )

        self.conv2 = nn.Conv2d(
            in_channels=20 * 20,
            out_channels=9 * 9,
            kernel_size=4,
            stride=2
        )

        self.conv3 = nn.Conv2d(
            in_channels=9 * 9,
            out_channels=7 * 7,
            kernel_size=3,
            stride=1
        )

        self.fc1 = nn.Linear(
            in_features=7 * 7,
            out_features=512
        )

        self.fc2 = nn.Linear(
            in_features=512,
            out_features=self.nb_action
        )

    def forward(self, frame_stake):
        x = torch.Tensor(frame_stake.flatten())
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
