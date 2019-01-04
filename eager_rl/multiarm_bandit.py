# language=rst
"""
Multi-arm bandit example from `Sutton and Barto <http://incompleteideas.net/book/bookdraft2017nov5.pdf>_`.
"""

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from eager_rl import History, eps_greedy
from tensorflow_probability import distributions as tfd

plt.ion()
tf.enable_eager_execution()


def main(k=2, method='max', timesteps=1000, distribution='normal', epsilon=0.05, plot=False):
    # Action, reward history.
    actions_taken = np.zeros(k)
    values = np.zeros(k)

    if distribution == 'normal':
        # Define unit normal and use it to sample means of reward distributions.
        standard_normal = tfd.Normal(loc=0, scale=1)
        means = standard_normal.sample(k)

        # Define normal distributions with unit normal sample means for rewards.
        reward_dist = tfd.Normal(loc=means, scale=tf.ones_like(means))
        optimal = tf.argmax(reward_dist.loc, dimension=0)
    else:
        raise NotImplementedError('Reward distribution not year implemented.')

    if plot:
        history = History(length=50)

    summed = 0
    maximum = 0
    for t in range(timesteps):
        # Sample reward observation for this episode.
        rewards = reward_dist.sample()

        if method == 'max':
            # Sample action (epsilon-greedy) and get reward.
            a = eps_greedy(values=values, epsilon=epsilon)
            r = rewards[a]

            # Update summed rewards.
            summed += r
            maximum += max(rewards)

            # Update history of taken actions and estimate of action values.
            actions_taken[a] += 1
            values[a] = values[a] + 1 / actions_taken[a] * (r - values[a])

            if plot:
                history.append(actions_taken[optimal] / (t + 1))
                history.plot()
    print()
    print(f'Total / maximum reward: {summed} / {maximum}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Multi-arm bandit.')
    parser.add_argument('-k', type=int, default=2, help='No. of bandit arms.')
    parser.add_argument('--method', type=str, default='max', help='Solution method ("max", ...)')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of episodes / observations.')
    parser.add_argument('--distribution', type=str, default='normal', help='Reward distribution.')
    parser.add_argument('--epsilon', type=float, default=0.05, help='Probability to select random arm.')
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.set_defaults(plot=False)
    args = vars(parser.parse_args())
    main(**args)
