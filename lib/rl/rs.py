import os

os.sys.path.insert(0, os.path.abspath("../.."))
import numpy as np
import pandas as pd

from lib.rl.memory import SequentialMemory

class RS(object):
    def __init__(self, nb_states, nb_actions, args):
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_actions = nb_actions
        self.is_training = True

        self.lbound = 0.  # args.lbound
        self.rbound = 1.  # args.rbound

        self.memory = SequentialMemory(limit = args.rmsize,
                                       window_length = args.window_length)

        self.run_id = args.run_id

        # self.design = pd.read_csv("experimental_designs/sobol_resnet50_600_samples.csv")
        self.design = pd.read_csv("current_design_{0}.csv".format(self.run_id))
        print(str(self.design))
        self.design["Top1"] = float("inf")
        self.design["Top5"] = float("inf")
        self.design["Size"] = float("inf")
        self.design["SizeRatio"] = float("inf")

        self.current_column = 0
        self.current_row = 0

        self.episode_end = False
        self.current_action = float("inf")

        self.double_bits = False

    def update_policy(self):
        pass

    def eval(self):
        pass

    def cuda(self):
        pass

    def observe(self, r_t, s_t, s_t1, a_t, done):
        pass

    def random_action(self):
        print("current_row: {}".format(self.current_row))
        print("current_column: {}".format(self.current_column))
        print("design shape: [rows: {0}, cols: {1}]".format(self.design.shape[0],
                                                            self.design.shape[1]))

        if self.double_bits:
            print("accessed rows: [{0}, {1}], [{2}, {3}]".format(self.current_row,
                                                                 self.current_column,
                                                                 self.current_row,
                                                                 self.current_column + 1))

            self.current_action = [self.design.iat[self.current_row,
                                                   self.current_column],
                                   self.design.iat[self.current_row,
                                                   self.current_column + 1]]

            self.current_column += 2
            if self.current_column >= self.design.shape[1] - 4:
                self.current_column = 0
                self.current_row += 1
        else:
            print("accessed rows: [{0}, {1}]".format(self.current_row,
                                                     self.current_column))

            self.current_action = self.design.iat[self.current_row, self.current_column]

            self.current_column += 1
            if self.current_column >= self.design.shape[1] - 4:
                self.current_column = 0
                self.current_row += 1

        return self.current_action

    def select_action(self, s_t, episode, decay_epsilon = True):
        print("episode: {0}".format(episode))
        if self.episode_end:
            self.episode_end = False
            return self.current_action
        else:
            self.past_episode = episode
            return self.random_action()

    def save_accuracy(self, top1, top5, size, size_ratio):
        self.episode_end = True
        self.design.at[self.current_row - 1, "Top1"] = top1
        self.design.at[self.current_row - 1, "Top5"] = top5
        self.design.at[self.current_row - 1, "Size"] = size
        self.design.at[self.current_row - 1, "SizeRatio"] = size_ratio
        self.design.to_csv("current_results_{0}.csv".format(self.run_id),
                           index = False)

    def reset(self, obs):
        pass

    def load_weights(self, output):
        pass

    def save_model(self, output):
        pass

    def seed(self, s):
        pass

    def soft_update(self, target, source):
        pass

    def hard_update(self, target, source):
        pass

    def get_delta(self):
        return np.array([0.0])

    def get_value_loss(self):
        return np.array([0.0])

    def get_policy_loss(self):
        return np.array([0.0])
