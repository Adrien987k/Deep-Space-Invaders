
import gym

import torch
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dq_net = models.DDDQNet(parameters.stack_size, parameters.nb_actions)
dq_net = dq_net.to(device)

#Fixed q-targets
target_net = models.DDDQNet(parameters.stack_size, parameters.nb_actions)
target_net = dq_net.to(device)

optimizer = optim.Adam(dq_net.parameters(), lr=parameters.learning_rate)

# NOT WORKING FOR NOW !!!
dq_net_trained, target_net_trained = train.train(
    dq_net, target_net, env, parameters, image_processor, actions, optimizer, device)

# test.test(dq_net_trained, env, actions, parameters, image_processor)
