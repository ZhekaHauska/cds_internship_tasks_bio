import numpy as np
from .metrics import MetricsRack, BaseLogger
from .scenario import Scenario
from typing import Any


class BaseAgent:
    logger: BaseLogger | None
    initial_action: int | None
    state_value: float
    true_state: Any

    def observe(self, events, action, reward=0):
        raise NotImplementedError

    def sample_action(self):
        raise NotImplementedError

    def reinforce(self, reward):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class BaseEnvironment:
    raw_obs_shape: (int, int)
    actions: tuple
    n_actions: int
    true_state: Any

    def obs(self):
        raise NotImplementedError

    def act(self, action):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def change_setup(self, setup):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class BaseRunner:
    agent: BaseAgent
    environment: BaseEnvironment
    metrics_rack: MetricsRack | None

    def __init__(self, logger, conf):
        """
        config_structure:
        run:
            n_episodes
            update_start
            max_steps
            reward_free
            action_inertia
            frame_skip
            strategies
            setups
        env:
            ,,,
        agent:
            ...
        metrics:
            ...
        """
        self.logger = logger
        self.seed = conf['run'].get('seed')
        self._rng = np.random.default_rng(self.seed)

        self.reward_free = conf['run'].get('reward_free', False)
        self.action_inertia = conf['run'].get('action_inertia', 1)
        self.frame_skip = conf['run'].get('frame_skip', 0)
        self.strategies = conf['run'].get('strategies', None)
        self.observe_actions = conf['run'].get('observe_actions', True)

        assert self.frame_skip >= 0

        self.current_setup_id = -1
        self.setup = conf['run']['setup']

        env_conf = conf['env']
        env_seed = env_conf.get('seed', None)
        if env_seed is None:
            env_conf['seed'] = self.seed
        agent_conf = conf['agent']
        agent_conf['seed'] = self.seed

        self.environment = self.make_environment(conf['env_type'], env_conf, self.setup)

        agent_conf['raw_obs_shape'] = self.environment.raw_obs_shape

        if self.observe_actions:
            agent_conf['n_actions'] = self.environment.n_actions
        else:
            agent_conf['n_actions'] = 0

        self.agent = self.make_agent(conf['agent_type'], agent_conf)

        metrics_conf = conf.get('metrics', None)
        if metrics_conf is not None and self.logger is not None:
            self.metrics_rack = MetricsRack(
                self.logger,
                self,
                **conf['metrics']
            )
        else:
            self.metrics_rack = None

        scenario_path = conf['run'].get('scenario', None)
        if scenario_path is not None:
            self.scenario = Scenario(scenario_path, self)
        else:
            self.scenario = None

        self.steps = 0
        self.total_steps = 0
        self.episodes = 0
        self.setup_episodes = 0
        self.strategy = None
        self.action_step = 0
        self.running = True
        self.action = self.agent.initial_action
        self.reward = 0
        self.episodic_reward = 0
        self.total_reward = 0
        self.events = None
        self.obs = None
        self.logging = False
        self.visualizing = False
        self.is_terminal = False
        self.end_of_episode = False

    @staticmethod
    def make_environment(env_type, conf, setup):
        raise NotImplementedError

    @staticmethod
    def make_agent(agent_type, conf):
        raise NotImplementedError

    def prepare_episode(self):
        self.steps = 0
        self.episodic_reward = 0
        self.is_terminal = False
        self.end_of_episode = False
        self.action = self.agent.initial_action

        self.environment.reset()
        self.agent.reset()

        self.strategy = None
        self.action_step = 0

    def run(self):
        self.episodes = 0
        self.setup_episodes = 0

        while self.running:
            self.prepare_episode()

            while not self.end_of_episode:
                if self.scenario is not None:
                    self.scenario.check_conditions()

                self.reward = 0
                self.obs = None
                for frame in range(self.frame_skip + 1):
                    if self.steps > 0:
                        self.environment.act(self.action)
                    self.environment.step()
                    self.obs, reward, self.is_terminal = self.environment.obs()
                    self.reward += reward
                    self.episodic_reward += reward
                    self.total_reward += reward

                    if self.is_terminal:
                        self.end_of_episode = True
                        break

                    if self.action is None:
                        break

                self.agent.true_state = self.environment.true_state
                # observe events_t, action_{t-1}, reward_{t}
                self.agent.observe(self.obs, self.action, self.reward)
                self.agent.reinforce(self.reward)

                if not self.end_of_episode:
                    if self.strategies is not None:
                        if self.steps == 0:
                            if self.reward_free:
                                strategy = self._rng.integers(len(self.strategies))
                            else:
                                strategy = self.agent.sample_action()

                            self.strategy = self.strategies[strategy]

                        if (self.steps % self.action_inertia) == 0:
                            if self.action_step < len(self.strategy):
                                self.action = self.strategy[self.action_step]
                            else:
                                self.end_of_episode = True
                            self.action_step += 1
                    else:
                        if (self.steps % self.action_inertia) == 0:
                            if self.reward_free:
                                self.action = self._rng.integers(self.environment.n_actions)
                            else:
                                self.action = self.agent.sample_action()

                if self.end_of_episode:
                    self.episodes += 1
                    self.setup_episodes += 1

                if (self.metrics_rack is not None) and self.logging:
                    self.metrics_rack.step()

                self.steps += 1
                self.total_steps += 1
        else:
            self.environment.close()
            if self.logger is not None:
                self.logger.close()

    def switch_logging(self):
        self.logging = not self.logging

    def switch_visualizing(self):
        self.visualizing = not self.visualizing

    def stop_episode(self):
        self.end_of_episode = True

    def stop_runner(self):
        self.running = False
        self.end_of_episode = True

    def change_setup(self, setup, setup_id):
        self.environment.change_setup(setup)
        self.environment.reset()

        self.setup = setup
        self.current_setup_id = setup_id
        self.setup_episodes = 0

    def set_parameters(self, **kwargs):
        def get_obj(path):
            obj = self
            for a in path:
                obj = getattr(obj, a)
            return obj

        for path, value in kwargs.items():
            path = path.split('.')
            att = path[-1]
            path = path[:-1]
            obj = get_obj(path)
            setattr(obj, att, value)
