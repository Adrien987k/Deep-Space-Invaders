
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from skimage import transform  # Help us to preprocess the frames
from skimage.color import rgb2gray  # Help us to gray our frames
from collections import deque
import environment

env, actions = environment.build_env()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class StackProcessor(object):

    def __init__(self, stack_size):
        self.stack_size = stack_size
        self.frame_stack = deque([np.zeros((110, 84), dtype=np.int)
                                  for i in range(self.stack_size)], maxlen=4)

        self.screen_height = 110
        self.screen_width = 84

    def init_stack(self, frame):
        self.frame_stack = deque([np.zeros((110, 84), dtype=np.int)
                                  for i in range(self.stack_size)], maxlen=4)

        for _ in range(self.stack_size):
            self.frame_stack.append(frame)

    def preprocess_frame(self, frame):

        # Greyscale the frame
        gray_frame = rgb2gray(frame)

        #Crop the screen (remove the part below the player)
        # [Up: Down, Left: right]
        cropped_frame = gray_frame[8:-12, 4:-12]

        # Normalize Pixel Values
        normalized_frame = cropped_frame / 255.0

        # Resize
        # Thanks to Mikołaj Walkowiak
        preprocessed_frame = transform.resize(normalized_frame, [110, 84])

        return preprocessed_frame  # 110x84x1 frame

    def stack_frame(self, frame, new_episode):

        frame = self.preprocess_frame(frame)

        if new_episode:
            self.init_stack(frame)
        else:
            self.frame_stack.append(frame)

        return torch.tensor(np.stack(self.frame_stack, axis=2))

stack_proc = StackProcessor(4)

class DQN(nn.Module):

    def __init__(self, stack_size, nb_actions):

        self.stack_size = stack_size
        self.nb_actions = nb_actions

        super(DQN, self).__init__()

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
            # TODO adapt to the stack size, also on the ameliorated versions ...
            in_features=64 * 5 * 4,
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

# class DQN(nn.Module):
#
#     def __init__(self, h, w):
#         super(DQN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
#         self.bn3 = nn.BatchNorm2d(32)
#
#         # Number of Linear input connections depends on output of conv2d layers
#         # and therefore the input image size, so compute it.
#         def conv2d_size_out(size, kernel_size = 5, stride = 2):
#             return (size - (kernel_size - 1) - 1) // stride  + 1
#
#         convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
#         convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
#         linear_input_size = convw * convh * 32
#         self.head = nn.Linear(linear_input_size, len(actions)) # 448 or 512
#
#     # Called with either one element to determine next action, or a batch
#     # during optimization. Returns tensor([[left0exp,right0exp]...]).
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         return self.head(x.view(x.size(0), -1))


############### Training
BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

N = len(actions)

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()


steps_done = 0
def select_action(policy_net, state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest value for column of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return  policy_net(state.view((1, stack_proc.stack_size, stack_proc.screen_height, stack_proc.screen_width))).max(1)[1].view(1, 1)

    else:
        return torch.tensor([[random.randrange(N)]], device=device, dtype=torch.long)

episode_durations = []

INF = float('inf')

def optimize_model(policy_net, memory):

    if len(memory) < BATCH_SIZE:
        return INF

    transitions = memory.sample(BATCH_SIZE)

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net

    state_action_values = policy_net(state_batch.view((BATCH_SIZE, stack_proc.stack_size, stack_proc.screen_height, stack_proc.screen_width)))
    state_action_values = state_action_values.gather(1, action_batch)


    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.

    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    #TODOOOOOOOOOOOOOOOOO EST-CE QUE CA A DU SENS DE FAIRE CA ????
    # Je veut dire le view ?
    # Ou alors il fallait en fait mettre tout bien avant de faire les trucs chelous...

    # TODO : Ca ne fonctionne pas, je fait un hack pour que ça passe mais il faut y reflechir avec Adrien....

    sss = non_final_next_states.size()
    sss = int(sss[0]/ stack_proc.screen_height)
    a =  target_net(non_final_next_states.view((sss, stack_proc.stack_size, stack_proc.screen_height, stack_proc.screen_width)))
    b = a.max(1)[0].detach()
    next_state_values[non_final_mask] = b.type(torch.FloatTensor)

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    c = expected_state_action_values.unsqueeze(1)

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, c.type(torch.DoubleTensor))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss

##########

class ModelsManager():

    def __init__(self):

        path = "models/saves/"
        self.simple_dqn_save_path = path + 'simple'

    def save_DQN_model(self, dq_net):
        torch.save(dq_net.state_dict(), self.simple_dqn_save_path)

    def load_DQN_model(self, stack_size,nb_actions, get_saved, device = 'cpu'):

        dq_net = DQN(stack_proc.stack_size, N).double()

        if get_saved:
            dq_net.load_state_dict(torch.load(self.simple_dqn_save_path, map_location = device))
            dq_net.eval()

        return dq_net

manager = ModelsManager()

policy_net = manager.load_DQN_model(stack_proc.stack_size, N, True)
target_net = DQN(stack_proc.stack_size, N).double()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


##################### MAIN TRAINING LOOP
#####################


def one_hot(action_index):
    return torch.eye(N)[action_index.item()]

num_episodes = 10
for i_episode in range(num_episodes):
    # Initialize the environment and state
    frame = env.reset()
    stack_proc = StackProcessor(4)
    state = stack_proc.stack_frame(frame, True)
    loss = INF

    for t in count():

        print('Episode: {} |> {}'.format(i_episode, t),
              'Training Loss {:.12f}'.format(loss))

        if t % 200 == 0:
            manager.save_DQN_model(policy_net)

        # Select and perform an action
        action = select_action(policy_net, state)

        next_state, reward, done, _ = env.step(one_hot(action))
        #env.render()

        reward = torch.tensor([reward], device=device)

        if not done:
            next_state = stack_proc.stack_frame(next_state, False)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        loss = optimize_model(policy_net, memory)

        if done:
            episode_durations.append(t + 1)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    manager.save_DQN_model(policy_net)

print('Complete')

import test

# Initialize the environment and state

stack_proc = StackProcessor(4)
test.test(policy_net, actions, env, stack_proc)

print('Goodbye')

env.close()
