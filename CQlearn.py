import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import gmean


ALPHA = 0.01
GAMMA = 0.9
EPS = 0.1
f = 0.99

A = np.array([0, 1.0, 1.0, 2.0, 2.0, 3.0])
B = np.array([0.5, 1.0, 1.0, 1.50, 1.50, 2.0])


def e_greedy(s, Q, EPS):
    if np.random.rand() > EPS:
        a_index = np.argmax(Q)
    else:
        a_index = np.random.randint(2)
    return a_index


def take_action(s, a_index):
    if a_index == 0:
        a = np.random.choice(A)
        s_ = s * (1.0 - f) + s * f * a
        reward = math.log(1.0 + (s_ / s - 1.0) * f)
    elif a_index == 1:
        a = np.random.choice(B)
        s_ = s * (1.0 - f) + s * f * a
        reward = math.log(1.0 + (s_ / s - 1.0) * f)
    return s_, reward


def do_action(s, a_index):
    if a_index == 0:
        a = np.random.choice(A)
        s_ = s + 1.0 * a
        reward = s_ - s
    elif a_index == 1:
        a = np.random.choice(B)
        s_ = s + 1.0 * a
        reward = s_ - s
    return s_, reward


def display(rounds, s):
    if (rounds + 1) % 100 == 0:
        print "rounds : %d , score : %f" % (rounds, s)


def Qlearning(n_rounds, t_max):
    s0 = 100.0
    Q = [0., 0.]
    return_history2 = []
    return_history = []
    for i in xrange(n_rounds):
        s = s0
        for t in xrange(t_max):
            a_index = e_greedy(s, Q, EPS)
            s_, reward = take_action(s, a_index)
            Q[a_index] += ALPHA * (reward + GAMMA * np.max(Q) - Q[a_index])
            s = s_
        print "round %d: %f" % (i, reward)
        print Q

        s = s0
        a_index = np.argmax(Q)
        s_, _ = take_action(s, a_index)
        r = (s_ / s - 1)
        s = s_
        x = (1 + r)
        return_history.append(x)
        return_history2.append(gmean(return_history))

    plt.plot(return_history2)
    plt.show()

if __name__ == '__main__':
    Qlearning(10000, 500)
