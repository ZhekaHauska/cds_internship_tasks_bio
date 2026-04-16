import numpy as np
from .base import BaseAgent
from .utils import EPS, normalize


class RandomAgent(BaseAgent):
    def __init__(self, raw_obs_shape, n_actions, seed):
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)
        self.n_actions = n_actions
        self.n_obs_states = raw_obs_shape[-1]
        self.initial_action = 0

        self.accuracy = 0
        self.surprise = 0
        self.true_state = None
        self.obs_state = -1
        # first order baseline
        self.first_order_transitions = np.zeros((n_actions, self.n_obs_states + 1, self.n_obs_states + 1))

    def observe(self, obs_state, action, reward=0):
        obs_state = obs_state[0]
        # first order baseline
        prediction = self.first_order_transitions[action][self.obs_state]
        prediction = normalize(prediction).squeeze()

        self.accuracy = float(np.argmax(prediction) == obs_state)
        self.surprise = - np.log(prediction[obs_state] + EPS)

        self.first_order_transitions[action][self.obs_state][obs_state] += 1
        self.obs_state = obs_state

    def sample_action(self):
        return self._rng.choice(self.n_actions)

    def reinforce(self, reward):
        ...

    def reset(self):
        self.accuracy = 0
        self.obs_state = -1
