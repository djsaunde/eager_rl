# language=rst
"""
Q-learning vs. Double Q-learning MDP example from
`Sutton and Barto <http://incompleteideas.net/book/bookdraft2017nov5.pdf>_`, page 110.
"""

import argparse
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from eager_rl import eps_greedy
from abc import ABC, abstractmethod
from tensorflow_probability import distributions as tfd

tf.enable_eager_execution()


class QLearningAgent:
    """
    Q-learning agent.
    """

    def __init__(self, states, actions, alpha=0.1, gamma=0.99):
        # Initialize Q values arbitarily.
        self.q_values = np.zeros([tf.size(states), tf.size(actions)])
        
        # Hyper-parameters.
        self.alpha = alpha
        self.gamma = gamma

    def update(self, s, a, r, s_):
        """
        Applies Q-learning update to Q values. That is, the TD(0) error is used to move the Q-values
        towards the target; i.e., $R_{t + 1} + \gamma \max_a Q(S_t, a)$.
        """
        self.q_values[s, a] += self.alpha * (
            r + self.gamma * tf.reduce_max(self.q_values[s_]) - self.q_values[s, a]
        )

class DoubleQLearningAgent:
    """
    Double Q-learning agent.
    """

    def __init__(self, states, actions, alpha=0.1, gamma=0.99):
        self.q_values1 = np.zeros([tf.size(states), tf.size(actions)])
        self.q_values2 = np.zeros([tf.size(states), tf.size(actions)])
        
        self.alpha = alpha
        self.gamma = gamma

    def update(self, s, a, r, s_):
        """
        Applies Double Q-learning update to Q values. That is, two sets of Q-values are estimated, and
        a coin is flipped to determine (1) which to update, and (2) which to use to estimate the maximum
        Q value. This mitigates the maximization bias of "vanilla" Q-learning (and similar methods).
        """
        if np.random.rand() < 0.5:
            self.q_values1[s, a] += self.alpha * (
                r + self.gamma * self.q_values2[s_, tf.argmax(self.q_values1[s_])] - self.q_values1[s, a]
            )
        else:
            self.q_values2[s, a] += self.alpha * (
                r + self.gamma * self.q_values1[s_, tf.argmax(self.q_values2[s_])] - self.q_values2[s, a]
            )


class MDP:
    """
    MDP from Example 6.7 in Sutton and Barto book (see above for link).
    """

    def __init__(self, k):
        self.k = k  # No. of actions from "B" into the left terminal state.
        
        self.states = tf.range(4)  # "A", "B", left terminal, right terminal
        self.actions = tf.range(2 + k)  # Left / right / "B" -> left terminal actions

        self.reward_dist = tfd.Normal(loc=-0.1, scale=1)  # Reward distribution from left terminal state

        # Transition kernel of MDP.
        self.transition_kernel = {
            (0, 0): 1, (0, 1): 3, **{(0, 2 + i): 0 for i in range(k)},
            (1, 0): 0, (1, 1): 1, **{(1, 2 + i): 2 for i in range(k)}
        }

        self.terminal = False

    def step(self, a):
        """
        Steps the MDP given the agent's choice of action ``a``, returns reward ``r``.
        """
        self.state = self.transition_kernel[(self.state, a)]

        if self.state in [2, 3]:
            self.terminal = True

        if self.state == 2:
            self.state = 0
            return 0
        
        elif self.state == 3:
            self.state = 0
            return tf.to_double(self.reward_dist.sample())

        return 0


def main(n=250, k=10, epsilon=0.1):
    mdp = MDP(k=k)
    mdp.state = 0

    agent1 = QLearningAgent(states=mdp.states, actions=mdp.actions)
    agent2 = DoubleQLearningAgent(states=mdp.states, actions=mdp.actions)
    
    cum_rewards1 = 0
    cum_rewards2 = 0
    
    for _ in tqdm(range(n)):
        while not mdp.terminal:
            s = mdp.state
            a = eps_greedy(agent1.q_values[s], epsilon=epsilon)
            r = mdp.step(a)
            s_ = mdp.state

            agent1.update(s, a, r, s_)

            cum_rewards1 += r
        
        mdp.terminal = False
        
        while not mdp.terminal:
            s = mdp.state
            a = eps_greedy(agent2.q_values1[s] + agent2.q_values2[s], epsilon=epsilon)
            r = mdp.step(a)
            s_ = mdp.state

            agent2.update(s, a, r, s_)

            cum_rewards2 += r
        
        mdp.terminal = False
    
    print(f'Cumulative rewards for Q-learning agent: {cum_rewards1.numpy()}')
    print(f'Cumulative rewards for Double Q-learning agent: {cum_rewards2.numpy()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=250, help='No. of episodes')
    parser.add_argument('-k', type=int, default=10, help='No. of actions that take the agent from state "B" into the left terminal state')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon of epsilon-greedy policy')
    args = parser.parse_args()
    args = vars(args)
    main(**args)
