import numpy as np
from matplotlib import pyplot
import random
import sys


class Bandit:

    def __init__(self, n_machine, n_action):
        for i in range(n_action):
            _qstar = np.random.normal(0.0, 1.0)
            _machine = np.random.normal(
                _qstar, 1.0, n_machine).reshape((-1, 1))
            if i == 0:
                self.machine = _machine
            else:
                self.machine = np.hstack((self.machine, _machine))

    def play(self, n_play, epsilon):
        self.q = np.zeros(self.machine.shape)
        self.q_count = np.zeros(self.machine.shape)
        average_reward = np.zeros(n_play)
        n_machine = self.machine.shape[0]

        for _p in range(n_play):
            total = 0.0
            for mac_index in range(n_machine):
                act_index = self.__select_action(mac_index, epsilon)
                reward = self.machine[mac_index, act_index]
                total += reward
                self.__update_qtable(reward, mac_index, act_index)

            average_reward[_p] = total / n_machine
            self.__display(_p, average_reward[_p])

        return average_reward

    def __select_action(self, mac_index, epsilon):
        if np.random.rand() > epsilon:
            act_index = self.__select_greedy_action(mac_index)
        else:
            act_index = self.__select_random_action()

        return act_index

    def __select_greedy_action(self, mac_index):
        _max = self.q[mac_index, :].max()
        indexes = np.argwhere(self.q[mac_index, :] == _max)
        random.shuffle(indexes)
        return indexes[0]

    def __select_random_action(self):
        return np.random.randint(10)

    def __update_qtable(self, reward, mac_index, act_index):
        _q = self.q[mac_index, act_index]
        self.q_count[mac_index, act_index] += 1
        self.q[mac_index, act_index] = _q + \
            (reward - _q) / self.q_count[mac_index, act_index]

    def __display(self, play, average_reward):
        if (play + 1) % 100 == 0:
            print u'play: %d, average reward: %f' % (play + 1, average_reward)
