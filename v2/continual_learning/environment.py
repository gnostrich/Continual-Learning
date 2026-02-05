"""Environment definitions for asynchronous continual learning."""

from collections import deque
import numpy as np
import torch


class BaseEnvironment:
    """Simple environment with task switching."""

    def __init__(
        self,
        observation_dim: int = 64,
        action_dim: int = 32,
        num_modalities: int = 2,
        task_difficulty: float = 0.5,
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.num_modalities = num_modalities
        self.task_difficulty = task_difficulty

        self.state = np.random.randn(observation_dim)
        self.time_step = 0
        self.current_task = 0
        self.task_params = np.random.randn(observation_dim)

    def reset(self):
        self.state = np.random.randn(self.observation_dim)
        self.time_step = 0
        return self.get_observation()

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        action_flat = action.mean(axis=0)
        if len(action_flat) < self.observation_dim:
            action_effect = np.zeros(self.observation_dim)
            action_effect[: len(action_flat)] = action_flat
        else:
            action_effect = action_flat[: self.observation_dim]

        self.state = 0.9 * self.state + 0.1 * action_effect
        self.state += 0.05 * self.task_params * self.task_difficulty
        self.state += np.random.randn(self.observation_dim) * 0.01

        self.time_step += 1
        observations = self.get_observation()
        reward = -np.linalg.norm(self.state - self.task_params)
        done = self.time_step >= 100
        info = {"time_step": self.time_step, "current_task": self.current_task}

        return observations, reward, done, info

    def get_observation(self):
        observations = []
        for idx in range(self.num_modalities):
            view = self.state + np.random.randn(self.observation_dim) * 0.1 * idx
            observations.append(torch.FloatTensor(view).unsqueeze(0))
        return observations

    def change_task(self, task_id: int = None):
        if task_id is not None:
            self.current_task = task_id
            rng = np.random.RandomState(task_id)
            self.task_params = rng.randn(self.observation_dim)
        else:
            self.current_task += 1
            self.task_params = np.random.randn(self.observation_dim)
        self.reset()


class DelayedObservationWrapper:
    """Wraps an environment to introduce observation delay."""

    def __init__(self, env: BaseEnvironment, delay_steps: int = 2):
        self.env = env
        self.delay_steps = max(0, delay_steps)
        self.queue = deque()

        self.observation_dim = env.observation_dim
        self.action_dim = env.action_dim
        self.num_modalities = env.num_modalities

    def reset(self):
        actual = self.env.reset()
        self.queue.clear()
        self.queue.append(actual)
        delayed = actual if self.delay_steps == 0 else None
        return delayed, actual

    def step(self, action):
        actual, reward, done, info = self.env.step(action)
        self.queue.append(actual)

        delayed = None
        if self.delay_steps == 0:
            delayed = actual
        elif len(self.queue) > self.delay_steps:
            delayed = self.queue.popleft()

        info = dict(info)
        info["delay_steps"] = self.delay_steps
        return delayed, actual, reward, done, info

    def change_task(self, task_id: int = None):
        self.env.change_task(task_id)
