
import models.nets as nets
import gym

import sys

import torch
import torch.optim as optim

import environment
import preprocess
import parameters
import train
import test
import models.models_manager as saver

import warnings
warnings.filterwarnings('ignore')

env, actions = environment.build_env()
parameters = parameters.Parameters(env, sys.argv)
image_processor = preprocess.ImageProcessor(env, actions, parameters)
models_manager = saver.ModelsManager(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dq_net, target_net = None, None

if parameters.simple_dqn:
    dq_net = models_manager.load_DQN_model(parameters, device)
else:
    dq_net, target_net = models_manager.load_DDDQN_model(parameters, device)

dq_net = dq_net.to(device)
target_net = dq_net.to(device)

optimizer = optim.SGD(dq_net.parameters(), lr=parameters.learning_rate)

if parameters.training:
    train.train(dq_net, target_net, env, parameters, image_processor,
                models_manager, actions, optimizer, device)

test.test(dq_net, env, actions, parameters, image_processor, device)
