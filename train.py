
import numpy as np
import random

import torch


def predict_action(model, parameters, decay_step, state, actions):
    """
    This function will do the part
    With ϵϵ select a random action atat, otherwise select at=argmaxaQ(st,a)
    """

    explore_stop = parameters.explore_stop
    explore_start = parameters.explore_start
    decay_rate = parameters.decay_rate

    # EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    # First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = parameters.explore_stop + \
        (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        choice = random.randint(1, len(actions)) - 1
        action = actions[choice]

    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        """ Qs = sess.run(DQNetwork.output, feed_dict={
                      DQNetwork.inputs_: state.reshape((1, *state.shape))}) """
        Qs = model(state)

        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = actions[choice]

    return action, explore_probability


def train(model, env, parameters, image_processor, actions, optimizer):

    # Initialize the decay rate (that will use to reduce epsilon)
    decay_step = 0

    for episode in range(parameters.total_episodes):
        # Set step to 0
        step = 0

        # Initialize the rewards of the episode
        episode_rewards = []

        # Make a new episode and observe the first state
        state = env.reset()

        # Remember that stack frame function also call our preprocess function.
        state = image_processor.stack_frame(state, True)

        while step < parameters.max_steps:
            step += 1

            # Increase decay_step
            decay_step += 1

            # Predict the action to take and take it
            action, explore_probability = predict_action(
                model, parameters, decay_step, state, actions)

            # Perform the action and get the next_state, reward, and done information
            next_state, reward, done, _ = env.step(action)

            if parameters.episode_render:
                env.render()

            # Add the reward to total reward
            episode_rewards.append(reward)

            # If the game is finished
            if done:
                # The episode ends so no next state
                next_state = np.zeros((110, 84), dtype=np.int)

                next_state = image_processor.stack_frame(
                    next_state, False)

                # Set step = max_steps to end the episode
                step = parameters.max_steps

                # Get the total reward of the episode
                total_reward = np.sum(episode_rewards)

                print('Episode: {}'.format(episode),
                      'Total reward: {}'.format(total_reward),
                      'Explore P: {:.4f}'.format(
                    explore_probability),
                    'Training Loss {:.4f}'.format(loss))

                # rewards_list.append((episode, total_reward))

                # Store transition <st,at,rt+1,st+1> in memory D
                image_processor.memory.add(
                    (state, action, reward, next_state, done))

            else:
                # Stack the frame of the next_state
                next_state = image_processor.stack_frame(
                    next_state, False)

                # Add experience to memory
                image_processor.memory.add(
                    (state, action, reward, next_state, done))

                # st+1 is now our current state
                state = next_state

            # LEARNING PART
            # Obtain random mini-batch from memory
            batch = image_processor.memory.sample(parameters.batch_size)
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            next_states_mb = np.array(
                [each[3] for each in batch], ndmin=3)
            dones_mb = np.array([each[4] for each in batch])

            target_Qs_batch = []

            # Get Q values for next_state
            Qs_next_state = model(next_states_mb)

            # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
            for i in range(0, len(batch)):
                terminal = dones_mb[i]

                # If we are in a terminal state, only equals reward
                if terminal:
                    target_Qs_batch.append(rewards_mb[i])

                else:
                    target = rewards_mb[i] + \
                        parameters.gamma * np.max(Qs_next_state[i])
                    target_Qs_batch.append(target)

            targets_mb = np.array([each for each in target_Qs_batch])

            Qs = model(states_mb)

            # Q is our predicted Q value.
            Q = (Qs * actions_mb).sum()

            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            loss = (torch.mul(targets_mb - Q, targets_mb - Q)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
