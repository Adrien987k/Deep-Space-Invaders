
import numpy as np
import torch

from itertools import count

def test(net, actions, env, stack_proc):

    N = len(actions)

    def one_hot(action_index):
        return torch.eye(N)[action_index.item()]

    frame = env.reset()
    state = stack_proc.stack_frame(frame, True)

    for t in count():

        with torch.no_grad():
            action = net(state.view((1, stack_proc.stack_size, stack_proc.screen_height, stack_proc.screen_width))).max(1)[1].view(1, 1)
            next_state, reward, done, _ = env.step(one_hot(action))
            env.render()
            state = stack_proc.stack_frame(next_state, False)

            if done:
                break
