
import torch
import models.nets as nets


class ModelsManager():

    def __init__(self, collab):

        path = "models/saves/"

        if collab:
            path = "/content/drive/My Drive/Deep-Space-Invaders/" + path

        self.ddqn_dq_path = path + 'ddqn_dq'
        self.ddqn_target_path = path + 'ddqn_target'
        self.simple_dqn_save_path = path + 'simple'

    def save_DDDQN_model(self, dq_net, target_net):

        torch.save(dq_net.state_dict(), self.ddqn_dq_path)
        torch.save(target_net.state_dict(), self.ddqn_target_path)

    def load_DDDQN_model(self, parameters, device='gpu'):

        dq_net = nets.DDDQNet(
            parameters.stack_size, parameters.nb_actions)
        target_net = nets.DDDQNet(
            parameters.stack_size, parameters.nb_actions)

        if parameters.get_saved_model:
            dq_net.load_state_dict(torch.load(
                self.ddqn_dq_path, map_location=device))
            target_net.load_state_dict(torch.load(
                self.ddqn_target_path, map_location=device))

            dq_net.eval()
            target_net.eval()

        return dq_net, target_net

    def save_DQN_model(self, dq_net, target_net):
        torch.save(dq_net.state_dict(), self.simple_dqn_save_path)

    def load_DQN_model(self, parameters, device='gpu'):

        dq_net = nets.SimpleDQNet(
            parameters.stack_size, parameters.nb_actions)

        if parameters.get_saved_model:
            dq_net.load_state_dict(torch.load(
                self.simple_dqn_save_path, map_location=device))
            dq_net.eval()

        return dq_net
