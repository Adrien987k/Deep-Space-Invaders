
import numpy as np
import torch

import random

def test(net, env, actions, parameters, image_processor, device):

    total_test_rewards = []

    for episode in range(1):
        total_rewards = 0

        state = env.reset()
        state = image_processor.stack_frame(state, True)

        print("****************************************************")
        print("EPISODE ", episode)

        while True:
            state = torch.Tensor(state).to(device)
            # Get action from Q-network or explore with low probability

            explore_probability = np.random.rand()

            if (explore_probability > 0.1):
                # Make a random action (exploration)
                choice = random.randint(1, len(actions)) - 1
                action = actions[choice]
            else:
                # Estimate the Qs values state
                Qs = net(state.view(1, 4, 110, 84))

                # Take the biggest Q value (= the best action)
                choice = np.argmax(Qs.detach().cpu().numpy())
                action = actions[choice]

            # print('Action =', action)

            # Perform the action and get the next_state, reward, and done information
            next_state, reward, done, _ = env.step(action)
            env.render()

            total_rewards += reward

            if done:
                print("Score", total_rewards)
                total_test_rewards.append(total_rewards)
                break

            next_state = image_processor.stack_frame(next_state, False)

            state = next_state

    env.close()
