# language=rst
"""
Multi-arm bandit example from `Sutton and Barto <http://incompleteideas.net/book/bookdraft2017nov5.pdf>_.
"""

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow_probability import distributions as tfd

tf.enable_eager_execution()


def main(k=2, method='max', timesteps=1000, distribution='normal', epsilon=0.05):
    # Action, reward history.
    actions_taken = np.zeros(k)
    sum_rewards_by_action = np.zeros(k)

    if distribution == 'normal':
        # Define unit normal and use it to sample means of reward distributions.
        standard_normal = tfd.Normal(loc=0, scale=1)
        means = standard_normal.sample(k)

        # Define normal distributions with unit normal sample means for rewards.
        reward_dist = tfd.Normal(loc=means, scale=tf.ones_like(means))
    else:
        raise NotImplementedError('Reward distribution not year implemented.')

    for t in range(timesteps):
        # Sample reward observation for this episode.
        rewards = reward_dist.sample()

        if method == 'max':
            # Compute values by taking sample mean over history of rewards.
            values = sum_rewards_by_action / actions_taken

            # Sample action (epsilon-greedy).
            if np.random.rand() < epsilon:
                # Randomly sample arm.
                a = np.random.choice(k)
            else:
                # Greedily choose arm with largest value estimate.
                a = tf.arg_max(values, dimension=0)

            # Update history of observed (action, reward) pairs.
            actions_taken[a] += 1
            sum_rewards_by_action[a] += rewards[a]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Multi-arm bandit.')
    parser.add_argument('-k', type=int, default=2, help='No. of bandit arms.')
    parser.add_argument('--method', type=str, default='max', help='Solution method ("max", ...)')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of episodes / observations.')
    parser.add_argument('--distribution', type=str, default='normal', help='Reward distribution.')
    parser.add_argument('--epsilon', type=float, default=0.05, help='Probability to select random arm.')
    args = vars(parser.parse_args())
    main(**args)
