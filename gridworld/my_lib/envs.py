import numpy as np
from copy import copy
import pygame
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import os
from .base import BaseEnvironment
from .utils import read_config


class GridWorld:
    def __init__(
            self,
            room,
            default_reward=0,
            observation_radius=0,
            collision_hint=False,
            collision_reward=0,
            headless=True,
            random_floor_colors=False,
            n_random_colors=0,
            markov_radius=0,
            blackout_prob=(0, 0),
            seed=None,
    ):
        self._rng = np.random.default_rng(seed)
        # TODO add separate obstacles layer
        room = np.asarray(room)
        self.colors, self.rewards, self.terminals, self.landmarks = (
            room[0, :, :], room[1, :, :], room[2, :, :], room[3, :, :]
        )

        self.random_floor_colors = random_floor_colors
        if self.random_floor_colors:
            # last color reserved for terminal state
            # negative colors reserved for obstacles
            if markov_radius == 0:
                colors = np.tile(
                    np.arange(n_random_colors),
                    (
                        int(self.colors.size//n_random_colors) +
                        int((self.colors.size % n_random_colors) > 0)
                    )
                )
                self._rng.shuffle(colors)
                colors = colors[:self.colors.size].reshape(self.colors.shape)
            else:
                colors = generate_map(markov_radius, self.colors.shape, seed)

            floor_mask = self.colors >= 0
            self.colors[floor_mask] = colors[floor_mask]

        self.landmark_colors = np.max(self.colors) + np.arange(np.count_nonzero(self.landmarks)) + 1
        self.colors[self.landmarks == 1] = self.landmark_colors

        self.h, self.w = self.colors.shape

        self.return_state = observation_radius < 0
        self.observation_radius = observation_radius
        self.collision_hint = collision_hint
        self.collision_reward = collision_reward
        self.blackout_prob = blackout_prob  # (on, off)
        self.headless = headless

        self.shift = max(self.observation_radius, 1)

        self.colors = np.pad(
            self.colors,
            self.shift,
            mode='constant',
            constant_values=-1
        ).astype(np.int32)

        self.unique_colors = np.unique(self.colors)

        if (not self.collision_hint) and (self.observation_radius <= 0):
            self.unique_colors = self.unique_colors[self.unique_colors >= 0]

        self.n_colors = len(self.unique_colors)
        print(f'[Gridworld] {self.w}x{self.h} map with {self.n_colors} colors.')

        if not self.return_state:
            self.observation_shape = (2*self.observation_radius + 1, 2*self.observation_radius + 1)
        else:
            self.observation_shape = (2,)

        self.start_r = None
        self.start_c = None
        self.r = None
        self.c = None
        self.action = None
        self.action_success = None
        self.temp_obs = None
        self.blackout = False
        # left, right, up, down
        self.actions = {0, 1, 2, 3}
        self.default_reward = default_reward
        self.time = 0

        if not self.headless:
            self.fig, self.ax = plt.subplots()
            window_size = self.render().size
            pygame.init()
            self.canvas = pygame.display.set_mode(window_size)
            pygame.display.set_caption('Gridworld')
            self.info = pygame.font.SysFont('hack', 14)
        else:
            self.info = None
            self.canvas = None
            self.fig = None
            self.ax = None

    def reset(self, start_r=None, start_c=None):
        if (start_r is None) or (start_c is None):
            while True:
                start_r = self._rng.integers(self.h)
                start_c = self._rng.integers(self.w)
                if self.colors[start_r + self.shift, start_c + self.shift] >= 0:
                    if self.terminals[start_r, start_c] == 0:
                        break
        else:
            assert self.colors[start_r + self.shift, start_c + self.shift] >= 0

        self.start_r, self.start_c = start_r, start_c
        self.r, self.c = start_r, start_c

        self.blackout = False
        self.temp_obs = None
        self.action = None
        self.time = 0

    def obs(self):
        assert self.r is not None
        assert self.c is not None

        obs = []

        if self.blackout:
            obs.append(None)
        else:
            if self.return_state:
                obs.append((self.r, self.c))
            else:
                if self.temp_obs is not None:
                    obs.append(copy(self.temp_obs))
                    self.temp_obs = None
                else:
                    obs.append(self._get_obs(self.r, self.c))

        reward = self.rewards[self.r, self.c] + self.default_reward
        if not self.action_success:
            reward += self.collision_reward

        obs.append(reward)
        obs.append(bool(self.terminals[self.r, self.c]))

        return obs

    def act(self, action):
        assert action in self.actions
        self.action = action

    def step(self):
        self.time += 1
        if self.action is not None:
            assert self.r is not None
            assert self.c is not None

            prev_r = self.r
            prev_c = self.c

            if self.action == 0:
                self.c -= 1
            elif self.action == 1:
                self.c += 1
            elif self.action == 2:
                self.r -= 1
            elif self.action == 3:
                self.r += 1

            # Check whether action is taking to inaccessible states.
            temp_x = self.colors[self.r+self.shift, self.c+self.shift]
            if temp_x < 0:
                self.r = prev_r
                self.c = prev_c

                if (not self.return_state) and self.collision_hint:
                    self.temp_obs = np.full(self.observation_shape, fill_value=temp_x)

                self.action_success = False
            else:
                self.action_success = True

        blackout_start, blackout_end = self.blackout_prob
        if blackout_start > 0:
            gamma = self._rng.random()
            if self.blackout:
                self.blackout = gamma > blackout_end
            else:
                self.blackout = gamma < blackout_start

        if self.canvas is not None:
            im = self.render()
            text = self.info.render(f'{self.blackout=}, {self.time=}', True, (0, 0, 0), (255, 255, 255))
            self.canvas.blit(pygame.image.fromstring(im.tobytes(), im.size, im.mode), (0, 0))
            self.canvas.blit(text, (0, 0))
            pygame.display.update()
            del im

    def _get_obs(self, r, c):
        r += self.shift
        c += self.shift
        start_r, start_c = r - self.observation_radius, c - self.observation_radius
        end_r, end_c = r + self.observation_radius + 1, c + self.observation_radius + 1
        obs = self.colors[start_r:end_r, start_c:end_c]
        return obs

    @property
    def colors_no_borders(self):
        if self.shift > 0:
            return self.colors[self.shift:-self.shift, self.shift:-self.shift]
        else:
            return self.colors

    def render(self):
        im = self.colors_no_borders.copy()
        min_vis_color = np.min(self.colors)

        if self.fig is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax.clear()

        # TODO automatically infer palette from number of colors
        sns.heatmap(im, annot=True, cmap='tab20', square=True, vmin=min_vis_color, cbar=False, ax=self.ax)

        if (self.r is not None) and (self.c is not None):
            self.ax.text(self.c, self.r+1, 'A', size='x-large')

        for s in np.flatnonzero(self.terminals):
            self.ax.text(s % self.w, s // self.w + 1, 'G', size='x-large')

        self.ax.axis('off')
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', bbox_inches="tight")
        plt.close('all')
        buf.seek(0)
        im = Image.open(buf)
        return im

    def get_true_map(self):
        true_map = self.colors_no_borders.copy()

        for i, color in enumerate(self.unique_colors):
            true_map[true_map == color] = i

        return true_map

    def get_true_matrices(self):
        n_states = self.h * self.w
        d_a = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        T = np.zeros((len(self.actions), n_states, n_states))
        E = np.zeros((n_states, len(self.unique_colors)))
        for a in self.actions:
            d_r, d_c = d_a[a]
            for r in range(self.h):
                for c in range(self.w):
                    x = self.colors[r + self.shift, c + self.shift]
                    if x < 0:
                        continue
                    state = c + r * self.w
                    E[state, np.flatnonzero(self.unique_colors == x)] = 1
                    r_next = r + d_r
                    c_next = c + d_c
                    # Check whether action is taking to inaccessible states.
                    next_x = self.colors[r_next + self.shift, c_next + self.shift]
                    if next_x < 0:
                        next_state = c + r * self.w
                    else:
                        next_state = c_next + r_next * self.w
                    T[a, state, next_state] = 1
        return T, E


class GridWorldWrapper(BaseEnvironment):
    environment: GridWorld
    def __init__(self, conf, setup):
        self.start_position = (None, None)
        self.conf = conf
        self.environment = self._start_env(setup)
        self.n_colors = self.environment.n_colors
        self.max_color = np.max(self.environment.unique_colors)
        self.min_color = np.min(self.environment.unique_colors)
        self.min_vis_color = np.min(self.environment.colors)
        self.trajectory = []
        self.is_first_step = True
        self.state_activity = np.zeros(self.environment.h * self.environment.w)
        self.activity_lr = 0.01

        self.n_cells = (
                (self.environment.observation_radius * 2 + 1) ** 2
        )

        if self.environment.return_state:
            self.raw_obs_shape = (1, self.environment.h * self.environment.w)
        else:
            self.raw_obs_shape = (
                self.n_cells,
                self.max_color - self.min_color + 1
            )
        self.actions = tuple(self.environment.actions)
        self.n_actions = len(self.actions)

    def obs(self):
        obs, reward, is_terminal = self.environment.obs()
        if obs is not None:
            if self.environment.return_state:
                obs = [obs[1] + obs[0]*self.environment.w]
            else:
                obs = obs.flatten()
                obs += (
                    np.arange(self.n_cells)*self.n_colors - self.min_color
                )
        else:
            obs = [None] * self.raw_obs_shape[0]

        self.is_first_step = False
        return obs, reward, is_terminal

    def act(self, action):
        if action is not None:
            gridworld_action = self.actions[action]
            self.environment.act(gridworld_action)

    def step(self):
        self.environment.step()

        self.trajectory.append(self.state)

        state = self.environment.w * self.environment.r + self.environment.c
        self.state_activity -= self.activity_lr * self.state_activity
        self.state_activity[state] += self.activity_lr

    def reset(self):
        self.environment.reset(*self.start_position)
        self.is_first_step = True
        self.trajectory.clear()

    def change_setup(self, setup):
        self.environment = self._start_env(setup)

    def close(self):
        del self.environment

    def get_true_matrices(self):
        return self.environment.get_true_matrices()

    @property
    def true_state(self):
        return self.environment.c + self.environment.r*self.environment.w

    @property
    def render(self):
        return self.environment.render()

    @property
    def state(self):
        shift = self.environment.shift
        im = self.environment.colors.astype('float32')
        agent_color = max(self.environment.unique_colors) + 0.5

        if shift > 0:
            im = im[shift:-shift, shift:-shift]

        im[self.environment.r, self.environment.c] = agent_color
        return im

    def _start_env(self, setup):
        config = read_config(setup)
        if 'start_position' in config:
            self.start_position = config['start_position']
        else:
            self.start_position = (None, None)

        env = GridWorld(
                room=np.array(config['room']),
                **self.conf
        )

        return env


def generate_map(markov_radius: int, size: tuple[int, int], seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    colors = np.full(size, fill_value=-1, dtype=np.int32)

    for _ in range(2):
        for r in range(markov_radius, size[0] - markov_radius):
            for c in range(markov_radius, size[1] - markov_radius):
                start_r, start_c = max(0, r - markov_radius), max(0, c - markov_radius)
                end_r, end_c = r + markov_radius + 1, c + markov_radius + 1

                window = colors[start_r:end_r, start_c:end_c]
                shape = window.shape
                window = window.flatten()

                # remove duplicates
                unique, positions = np.unique(window, return_index=True)
                window = np.full_like(window, fill_value=-1)
                window[positions] = unique

                # fill empty space
                empty_mask = window == -1
                n_nonzero = np.count_nonzero(empty_mask)
                if n_nonzero == 0:
                    continue

                candidates = np.arange(window.size)
                candidates = candidates[np.isin(candidates, window, invert=True)]
                candidates = candidates[:n_nonzero]
                rng.shuffle(candidates)
                window[empty_mask] = candidates
                colors[start_r:end_r, start_c:end_c] = window.reshape(shape)
    return colors
