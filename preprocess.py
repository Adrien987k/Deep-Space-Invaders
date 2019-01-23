
import numpy as np           # Handle matrices

import random

from collections import deque  # Ordered collection with ends

from skimage import transform  # Help us to preprocess the frames
from skimage.color import rgb2gray  # Help us to gray our frames


class SumTree(object):
    """
    This SumTree code is modified version of Morvan Zhou:
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):

        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            else:

                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]


class PERMemory(object):

    PER_e = 0.01
    PER_a = 0.6
    PER_b = 0.4

    PER_b_increment_per_sampling = 0.001

    default_priority = 1.

    def __init__(self, max_size):
        self.tree = SumTree(max_size)

    def add(self, experience):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        if max_priority <= 0:
            max_priority = self.default_priority

        self.tree.add(max_priority, experience)

    def sample(self, batch_size):
        batch = []
        b_idx, b_ISWeights = np.empty((batch_size,), dtype=np.int32), np.empty(
            (batch_size, 1), dtype=np.float32)

        priority_segment = self.tree.total_priority / batch_size

        self.PER_b = np.min(
            [1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # TODO check if my hack is correct (@Lost)
        p_min = np.min([e for e in self.tree.tree[-self.tree.capacity:]
                        if e > 0]) / self.tree.total_priority
        max_weight = (p_min * batch_size) ** (-self.PER_b)

        for i in range(batch_size):

            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            index, priority, data = self.tree.get_leaf(value)

            sampling_probabilities = priority / self.tree.total_priority

            b_ISWeights[i, 0] = np.power(
                batch_size * sampling_probabilities, -self.PER_b) / max_weight

            b_idx[i] = index

            experience = [data]

            batch.append(experience)

        # batch = [e[0] for e in batch] #Eliminating a 1 in the dimension...
        return b_idx, np.squeeze(batch), b_ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.default_priority)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class Memory():
    def __init__(self, max_size):
        self.buffer = deque()
        self.max_size = max_size

    def add(self, experience):
        self.buffer.append(experience)

        if len(self.buffer) > self.max_size:
            self.buffer.popleft()

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)

        return [self.buffer[i] for i in index]


class ImageProcessor:

    def __init__(self, env, actions, parameters, per):

        self.stack_size = parameters.stack_size
        self.frame_stack = None

        if per:
            self.memory = PERMemory(max_size=parameters.memory_size)
        else:
            self.memory = Memory(max_size=parameters.memory_size)

        for i in range(parameters.pretrain_length):
            # If it's the first step
            if i == 0:
                state = env.reset()
                state = self.stack_frame(state, True)

            # Get the next_state, the rewards, done by taking a random action
            choice = random.randint(1, len(actions)) - 1
            action = actions[choice]
            next_state, reward, done, _ = env.step(action)

            # Stack the frames
            next_state = self.stack_frame(next_state, False)

            # If the episode is finished (we're dead 3x)
            if done:
                # We finished the episode
                next_state = np.zeros(state.shape)

                # Add experience to memory
                self.memory.add((state, action, reward, next_state, done))

                # Start a new episode
                state = env.reset()

                # Stack the frames
                state = self.stack_frame(state, True)

            else:
                # Add experience to memory
                self.memory.add((state, action, reward, next_state, done))

                # Our new state is now the next_state
                state = next_state

    def init_stack(self, frame):
        self.frame_stack = deque([np.zeros((110, 84), dtype=np.int)
                                  for i in range(self.stack_size)], maxlen=4)

        for _ in range(self.stack_size):
            self.frame_stack.append(frame)

    def preprocess_frame(self, frame):

        # Greyscale the frame
        gray_frame = rgb2gray(frame)

        # Crop the screen (remove the part below the player)
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

        return np.stack(self.frame_stack, axis=2)
