
import torch
import models.models as nets


class ModelsManager():

    def __init__(self):

        path = "models/saves/"
        self.simple_dqn_save_path = path + 'simple'


    def save_DQN_model(self, dq_net):
        torch.save(dq_net.state_dict(), self.simple_dqn_save_path)

    def load_DQN_model(self, parameters, device = 'gpu'):

        dq_net = nets.SimpleDQNet(
            parameters.stack_size, parameters.nb_actions)

        if parameters.get_saved_model:
            dq_net.load_state_dict(torch.load(self.simple_dqn_save_path, map_location = device))
            #target_net.load_state_dict(torch.load(self.ddqn_target_path, map_location = device))

            dq_net.eval()

        return dq_net
