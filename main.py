
import gym

import torch.optim as optim

import environment
import preprocess
import parameters
import train
import test
import models.models as models


env, actions = environment.build_env()
parameters = parameters.Parameters(env)
image_processor = preprocess.ImageProcessor(env, actions, parameters)

dq_net = models.SimpleDQNet(parameters.nb_states, parameters.nb_actions)

optimizer = optim.Adam(dq_net.parameters(), lr=parameters.learning_rate)

# NOT WORKING FOR NOW !!!
# dq_net_trained = train.train(
#     dq_net, env, parameters, image_processor, actions, optimizer)

# test.test(dq_net_trained, env, actions, parameters, image_processor)
