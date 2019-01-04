import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path

ROOT_DIR = Path(__file__).parents[0].parents[0]


class History:
    # language=rst
    """
    History buffer implementation.
    """

    def __init__(self, length):
        # language=rst
        """
        Constructor of ``History`` class.

        :param length: Length of history.
        """
        self.length = length
        self.buffer = []
        self.t = 0

        self.fig, self.ax = plt.subplots()

        plt.xlabel('Time steps')
        plt.ylabel('Estimated reward')
        plt.title('History of estimated rewards')

    def append(self, datum):
        # language=rst
        """
        Add to the history, respecting its length.

        :param datum: Datum to add to end of history.
        """
        if len(self.buffer) == self.length:
            self.buffer = self.buffer[1:] + [datum]
        else:
            self.buffer.append(datum)

        self.t += 1

    def plot(self, lines=None):
        # language=rst
        """
        Plot current state of history.
        """
        self.ax.clear()
        self.ax.plot(range(max(0, (self.t - 50)), self.t), self.buffer)

        if lines is not None:
            for line in lines:
                plt.axhline(line, linestyle='--')

        plt.ylim([0, 1])
        plt.xticks(range(max(0, ((self.t - 50) // 10 + 1) * 10), self.t, 10))
        plt.pause(0.1)


def greedy(values):
    # language=rst
    """
    Greedily choose maximally valued action.

    :param values: Estimated values of actions.
    :return: Greedily chosen action.
    """
    return tf.argmax(values, dimension=0)


def eps_greedy(values, epsilon=0.05):
    # language=rst
    """
    Greedily choose maximally valued action with proability ``1 - epsilon``; otherwise, choose an action uniformly at
    random.

    :param values: Estimated values of actions
    :param epsilon: Probability with which to choose an action uniformly at random.
    :return: Epsilon-greedily chosen action.
    """
    if np.random.rand() < epsilon:
        return np.random.choice(tf.size(values))
    return tf.argmax(values, dimension=0).numpy()


def softmax(preferences):
    # language=rst
    """
    Samples from softmax distribution over preferences of actions.

    :param preferences: The larger the preference, the greater the (relative) probability of selecting the action.
    :return: Action sampled from softmax distribution over preferences.
    """
    distribution = tf.exp(preferences) / tf.reduce_sum(tf.exp(preferences))
