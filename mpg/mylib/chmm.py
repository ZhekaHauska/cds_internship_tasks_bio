import numpy as np
from typing import Literal, Optional

# available initialization for transition matrix
INI_MODE = Literal['dirichlet', 'normal', 'uniform']


def softmax(x: np.ndarray, temp=1.) -> np.ndarray:
    """Computes softmax values for a vector `x` with a given temperature."""
    temp = np.clip(temp, 1e-5, 1e+3)
    e_x = np.exp((x - np.max(x, axis=-1)) / temp)
    return e_x / e_x.sum(axis=-1)


class CHMM:
    def __init__(
            self,
            n_columns: int,
            cells_per_column: int,
            lr: float = 0.1,
            batch_size: int = 1,
            initialization: INI_MODE = 'uniform',
            sigma: float = 1.0,
            alpha: float = 1.0,
            seed: Optional[int] = None
    ):
        """
        Custom realization of CHMM method from paper https://arxiv.org/abs/1905.00507.
        This is just a template, you can change this class as needed.

        n_columns:
        cells_per_columns: number of hidden state copies for an observation state, we also call them `columns`
        lr: learning rate for matrix updates
        batch_size: sequence size for learning
        initialization: transition matrix initialization
        sigma: parameter of normal distribution
        alpha: parameter of alpha distribution
        seed: seed for reproducibility, None means no reproducibility
        """

        self.n_columns = n_columns
        self.cells_per_column = cells_per_column
        self.n_states = cells_per_column * n_columns
        self.states = np.arange(self.n_states)
        self.lr = lr
        self.batch_size = batch_size
        self.initialization = initialization
        self.is_first = True
        self.seed = seed

        self._rng = np.random.default_rng(self.seed)

        if self.initialization == 'dirichlet':
            self.transition_probs = self._rng.dirichlet(
                alpha=[alpha]*self.n_states,
                size=self.n_states
            )
            self.state_prior = self._rng.dirichlet(alpha=[alpha]*self.n_states)
        elif self.initialization == 'normal':
            self.log_transition_factors = self._rng.normal(
                scale=sigma,
                size=(self.n_states, self.n_states)
            )
            self.log_state_prior = self._rng.normal(scale=sigma, size=self.n_states)
        elif self.initialization == 'uniform':
            self.log_transition_factors = np.zeros((self.n_states, self.n_states))
            self.log_state_prior = np.zeros(self.n_states)

        if self.initialization != 'dirichlet':
            self.transition_probs = np.vstack(
                [softmax(x) for x in self.log_transition_factors]
            )

            self.state_prior = softmax(self.log_state_prior)
        else:
            self.log_transition_factors = np.log(self.transition_probs)
            self.log_state_prior = np.log(self.state_prior)

    def observe(self, observation_state: int, learn: bool = True) -> None:
        """
        Here method gets new observation state and make matrix updates.
        """
        raise NotImplementedError

    def predict_observation_states(self) -> np.ndarray:
        """
        Should return probabilities of observation states for the next timestep, which sum to 1.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Here make preparations for starting new sequence of observations.
        """
        self.is_first = True
        raise NotImplementedError
