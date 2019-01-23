
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class Statistics():
    """
    Save statistics during training/testing
    Draw curves
    """

    def __init__(self):
        self.path = "plots/"

        self.episodes = []
        self.rewards = []
        self.explore_probabilities = []
        self.losses = []

    def add_episode_stats(self, episode, total_reward, explore_probability, loss):
        self.episodes.append(episode)
        self.rewards.append(total_reward)
        self.explore_probabilities.append(explore_probability)
        self.losses.append(loss)

        if self.episodes[-1] % 5 == 0 and self.episodes[-1] > 0:
            self.save_plots()

    def save_plot(self, title, xlabel, ylabel, values):

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.plot(np.array(self.episodes), np.array(values))
        plt.savefig(self.path + ylabel + str(self.episodes[-1]) + '.png')

    def save_plots(self):

        self.save_plot("Evolution of reward over episodes",
                       "Episodes", "Rewards", self.rewards)
        self.save_plot("Evolution of loss over episodes",
                       "Episodes", "Loss", self.losses)
        self.save_plot("Exploration probability over episodes",
                       "Episodes", "exploration_probability", self.explore_probabilities)
