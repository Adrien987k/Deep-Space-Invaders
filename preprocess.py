
import numpy as np           # Handle matrices

import random

from collections import deque  # Ordered collection with ends

from skimage import transform  # Help us to preprocess the frames
from skimage.color import rgb2gray  # Help us to gray our frames


class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)

        return [self.buffer[i] for i in index]


class ImageProcessor:

    def __init__(self, env, actions, parameters):

        self.stack_size = parameters.stack_size
        self.frame_stack = None
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

            # env.render()

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

        return np.stack(self.frame_stack, axis=2)
