"""Training loop for asynchronous continual learning."""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim


class ContinualLearningLoop:
    """Continual learning loop with explicit observation delay handling."""

    def __init__(
        self,
        model,
        predictor,
        environment,
        learning_rate: float = 3e-4,
        ewc_lambda: float = 0.5,
        feedback_weight: float = 0.3,
        divergence_weight: float = 1.0,
        action_l2_weight: float = 1e-3,
        observation_blend: float = 0.5,
    ):
        self.model = model
        self.predictor = predictor
        self.environment = environment
        self.learning_rate = learning_rate
        self.ewc_lambda = ewc_lambda
        self.feedback_weight = feedback_weight
        self.divergence_weight = divergence_weight
        self.action_l2_weight = action_l2_weight
        self.observation_blend = observation_blend

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.predictor_optimizer = optim.Adam(self.predictor.parameters(), lr=learning_rate)

        self.fisher_information = {}
        self.optimal_params = {}
        self.task_count = 0

    def compute_fisher_information(self, num_samples: int = 50):
        self.model.eval()
        fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)

        for _ in range(num_samples):
            self.model.zero_grad()
            self.model.reset_state()
            delayed_obs, _ = self.environment.reset()
            observations = delayed_obs
            if observations is None:
                observations = self._zero_observation(1, device=self._device())
            output, _ = self.model(observations)
            log_prob = -torch.norm(output, p=2)
            log_prob.backward()
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2

        for name in fisher:
            fisher[name] /= num_samples
        self.fisher_information = fisher

    def save_optimal_params(self):
        self.optimal_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()

    def ewc_loss(self):
        if not self.fisher_information:
            return torch.tensor(0.0, device=self._device())
        loss = torch.tensor(0.0, device=self._device())
        for name, param in self.model.named_parameters():
            if name in self.fisher_information and param.requires_grad:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                loss += (fisher * (param - optimal) ** 2).sum()
        return loss

    def compute_divergence(self):
        if not self.optimal_params:
            return 0.0
        total = 0.0
        count = 0
        for name, param in self.model.named_parameters():
            if name in self.optimal_params and param.requires_grad:
                optimal = self.optimal_params[name]
                total += torch.norm(param - optimal, p=2).item()
                count += 1
        return total / max(count, 1)

    def _device(self):
        for param in self.model.parameters():
            return param.device
        return torch.device("cpu")

    def _zero_observation(self, batch_size: int, device: torch.device):
        obs = []
        for _ in range(self.environment.num_modalities):
            obs.append(torch.zeros(batch_size, self.environment.observation_dim, device=device))
        return obs

    def _blend_observations(self, delayed, predicted):
        blended = []
        for d_obs, p_obs in zip(delayed, predicted):
            blended.append((1.0 - self.observation_blend) * d_obs + self.observation_blend * p_obs)
        return blended

    def train_step(self, action, internal_state, actual_response):
        self.model.train()
        self.predictor.train()
        self.optimizer.zero_grad()
        self.predictor_optimizer.zero_grad()

        predicted_responses, _ = self.predictor(action, internal_state)

        divergence_loss = torch.tensor(0.0, device=action.device)
        for pred, actual in zip(predicted_responses, actual_response):
            divergence_loss += nn.functional.mse_loss(pred, actual)

        ewc_reg = self.ewc_loss()
        action_reg = action.pow(2).mean()

        total_loss = (
            self.divergence_weight * divergence_loss
            + self.action_l2_weight * action_reg
            + self.ewc_lambda * ewc_reg
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.predictor_optimizer.step()

        if self.model.hidden_state is not None:
            self.model.hidden_state = self.model.hidden_state.detach()
        if self.predictor.hidden_state is not None:
            self.predictor.hidden_state = self.predictor.hidden_state.detach()

        return total_loss.item()

    def run_episode(self, max_steps: int = 100, verbose: bool = False):
        delayed_obs, actual_obs = self.environment.reset()
        self.model.reset_state()
        self.predictor.reset_state()

        episode_losses = []
        episode_rewards = []
        internal_state = None
        last_action = None
        last_internal = None

        for step in range(max_steps):
            if self.model.hidden_state is not None:
                self.model.hidden_state = self.model.hidden_state.detach()
            if self.predictor.hidden_state is not None:
                self.predictor.hidden_state = self.predictor.hidden_state.detach()

            predicted_current = None
            if last_action is not None and last_internal is not None:
                predicted_current, _ = self.predictor(last_action, last_internal)
                predicted_current = [p.detach() for p in predicted_current]

            if delayed_obs is None:
                if predicted_current is None:
                    observations = self._zero_observation(1, device=self._device())
                else:
                    observations = predicted_current
            else:
                if predicted_current is not None and self.observation_blend > 0:
                    observations = self._blend_observations(delayed_obs, predicted_current)
                else:
                    observations = delayed_obs

            external_feedback = (
                internal_state * self.feedback_weight if internal_state is not None else None
            )

            action, internal_state = self.model(observations, external_feedback)
            internal_state = internal_state.detach()

            delayed_obs, actual_obs, reward, done, info = self.environment.step(action.detach())
            episode_rewards.append(reward)

            loss = self.train_step(action, internal_state, actual_obs)
            episode_losses.append(loss)

            last_action = action.detach()
            last_internal = internal_state.detach()

            if done:
                break

        metrics = {
            "mean_loss": sum(episode_losses) / max(len(episode_losses), 1),
            "total_reward": sum(episode_rewards),
            "steps": len(episode_losses),
        }

        if verbose:
            print(
                "Episode: Loss={:.4f}, Reward={:.2f}, Steps={}".format(
                    metrics["mean_loss"], metrics["total_reward"], metrics["steps"]
                )
            )
        return metrics

    def continual_learning_cycle(
        self, num_tasks: int = 3, episodes_per_task: int = 10, verbose: bool = True
    ):
        all_metrics = {"task_losses": [], "task_rewards": [], "divergences": []}

        for task_id in range(num_tasks):
            if verbose:
                print("\n=== Task {}/{} ===".format(task_id + 1, num_tasks))

            self.environment.change_task(task_id)
            task_losses = []
            task_rewards = []

            for episode in range(episodes_per_task):
                metrics = self.run_episode(verbose=verbose and episode % 5 == 0)
                task_losses.append(metrics["mean_loss"])
                task_rewards.append(metrics["total_reward"])

            if verbose:
                print("Computing Fisher information for task {}...".format(task_id + 1))
            self.compute_fisher_information(num_samples=30)
            self.save_optimal_params()

            divergence = self.compute_divergence()
            all_metrics["task_losses"].append(task_losses)
            all_metrics["task_rewards"].append(task_rewards)
            all_metrics["divergences"].append(divergence)

            if verbose:
                print(
                    "Task {} complete. Avg Loss: {:.4f}, Avg Reward: {:.2f}, Divergence: {:.4f}".format(
                        task_id + 1,
                        sum(task_losses) / len(task_losses),
                        sum(task_rewards) / len(task_rewards),
                        divergence,
                    )
                )
            self.task_count += 1

        return all_metrics
