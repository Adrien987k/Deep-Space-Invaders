
import gym

import torch
import torch.optim as optim

import environment
import preprocess
import parameters
import train
import test
import models.models as models
import models.models_manager as saver


env, actions = environment.build_env()
parameters = parameters.Parameters(env)
image_processor = preprocess.ImageProcessor(env, actions, parameters)
model_manager = saver.ModelsManager()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dq_net = model_manager.load_DQN_model(parameters, device)
dq_net = dq_net.to(device)

optimizer = optim.Adam(dq_net.parameters(), lr=parameters.learning_rate)

# NOT WORKING FOR NOW !!!
dq_net_trained = train.train(
    dq_net, env, parameters, image_processor, model_manager, actions, optimizer, device)

test.test(dq_net_trained, env, actions, parameters, image_processor)
