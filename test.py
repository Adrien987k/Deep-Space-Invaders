
import numpy as np
import torch

from itertools import count

def test(net, actions, env, stack_proc):

    N = len(actions)

    def one_hot(action_index):
        return torch.eye(N)[action_index.item()]

    frame = env.reset()
    state = stack_proc.stack_frame(frame, True)

    print(env.action_set)
    print(N)

    for t in count():

        with torch.no_grad():
            a = net(state.view((1, stack_proc.stack_size, stack_proc.screen_height, stack_proc.screen_width)))
            action = a.max(1)[1].view(1, 1)
            print(action)
            print(a)
            action = torch.tensor([5])
            print(action)
            next_state, reward, done, _ = env.step(one_hot(action))
            env.render()
            state = stack_proc.stack_frame(next_state, False)

            if done:
                break
