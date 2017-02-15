from bandit import *

if __name__ == '__main__':

    # param
    param = sys.argv

    # init
    n_machine = 2000
    n_action = 10
    n_play = 2000
    epsilon = [0.0, 0.01, 0.1, 1.0]

    # draw init
    mergin = 5
    color = ['b', 'g', 'r', 'k']
    pyplot.figure(figsize=(8, 6))
    pyplot.xlim(-mergin, n_play + mergin)

    # bandit machine
    bandit = Bandit(n_machine, n_action)

    # play
    for i in range(len(epsilon)):
        print u'play count: %d, epsilon: %.2f' % (n_play, epsilon[i])
        average_reward = bandit.play(n_play, epsilon[i])
        _label = 'e = %.2f' % epsilon[i]
        pyplot.plot(np.arange(n_play), average_reward,
                    color=color[i], label=_label)
        print '!!!finish!!!\n'

    # save and show
    if '-d' in param or '-s' in param:
        pyplot.legend(loc='center right')
        if '-s' in param:
            pyplot.savefig('bandit2.png')
        if '-d' in param:
            pyplot.show()
