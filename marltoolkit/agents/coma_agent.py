import os
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
from torch.distributions import Categorical

from marltoolkit.agents.base_agent import BaseAgent
from marltoolkit.modules.actors.rnn import RNNActorModel
from marltoolkit.modules.critics.coma import MLPCriticModel
from marltoolkit.utils import (LinearDecayScheduler, MultiStepScheduler,
                               hard_target_update)


class ComaAgent(BaseAgent):
    """Coma algorithm
    Args:
        actor_model (nn.Model): agents' local q network for decision making.
        critic_model (nn.Model): A mixing network which takes local q values as input
            to construct a global Q network.
        double_q (bool): Double-DQN.
        td_lambda (float): lambda of td-lambda return
        gamma (float): discounted factor for reward computation.
        actor_lr (float): actor network learning rate
        critic_lr (float): critic network learning rate
        clip_grad_norm (None, or float): clipped value of gradients' global norm.
    """

    def __init__(
        self,
        actor_model: RNNActorModel = None,
        critic_model: MLPCriticModel = None,
        num_envs: int = 1,
        num_agents: int = None,
        n_actions: int = None,
        double_q: bool = True,
        total_steps: int = 1e6,
        gamma: float = 0.99,
        optimizer_type: str = 'rmsprop',
        td_lambda: float = 0.8,
        entropy_coef: float = 0.01,
        actor_lr: float = 0.0005,
        critic_lr: float = 0.0001,
        exploration_start: float = 1.0,
        min_exploration: float = 0.01,
        update_target_interval: int = 100,
        update_learner_freq: int = 1,
        clip_grad_norm: float = 10,
        optim_alpha: float = 0.99,
        optim_eps: float = 0.00001,
        device: str = 'cpu',
    ) -> None:
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.n_actions = n_actions
        self.double_q = double_q
        self.gamma = gamma
        self.optimizer_type = optimizer_type
        self.td_lambda = td_lambda
        self.entropy_coef = entropy_coef
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_grad_norm = clip_grad_norm
        self.global_steps = 0
        self.exploration = exploration_start
        self.min_exploration = min_exploration
        self.target_update_count = 0
        self.update_target_interval = update_target_interval
        self.update_learner_freq = update_learner_freq
        self.device = device

        self.actor_model = actor_model.to(device)
        self.target_actor_model = deepcopy(actor_model).to(device)

        self.critic_model = critic_model.to(device)
        self.target_critic_model = deepcopy(critic_model).to(device)

        self.actor_params = list(self.actor_model.parameters())
        self.critic_params = list(self.critic_model.parameters())

        if self.optimizer_type == 'adam':
            self.agent_optimiser = torch.optim.Adam(params=self.actor_params,
                                                    lr=self.actor_lr)
            self.critic_optimiser = torch.optim.Adam(params=self.critic_params,
                                                     lr=self.critic_lr)
        else:
            self.actor_optimizer = torch.optim.RMSprop(
                params=self.actor_params,
                lr=self.actor_lr,
                alpha=optim_alpha,
                eps=optim_eps,
            )

            self.critic_optimizer = torch.optim.RMSprop(
                params=self.actor_params,
                lr=self.critic_lr,
                alpha=optim_alpha,
                eps=optim_eps,
            )

        self.ep_scheduler = LinearDecayScheduler(exploration_start,
                                                 total_steps * 0.8)

        lr_steps = [total_steps * 0.5, total_steps * 0.8]
        self.actor_lr_scheduler = MultiStepScheduler(
            start_value=self.actor_lr,
            max_steps=total_steps,
            milestones=lr_steps,
            decay_factor=0.5,
        )
        self.critic_lr_scheduler = MultiStepScheduler(
            start_value=self.critic_lr,
            max_steps=total_steps,
            milestones=lr_steps,
            decay_factor=0.5,
        )

        # 执行过程中，要为每个agent都维护一个 hidden_state
        # 学习过程中，要为每个agent都维护一个 hidden_state、target_hidden_state
        self.hidden_state = None
        self.target_hidden_state = None

    def init_hidden_states(self, batch_size: int = 1) -> None:
        """Initialize hidden states for each agent.

        Args:
            batch_size (int): batch size
        """
        self.hidden_state = self.actor_model.init_hidden()
        self.hidden_state = self.hidden_state.unsqueeze(0).expand(
            batch_size, self.num_agents, -1)

        self.target_hidden_state = self.target_actor_model.init_hidden()
        if self.target_hidden_state is not None:
            self.target_hidden_state = self.target_hidden_state.unsqueeze(
                0).expand(batch_size, self.num_agents, -1)

    def sample(self, obs: np.array, available_actions: np.array):
        """sample actions via epsilon-greedy
        Args:
            obs (np.ndarray):               (num_agents, obs_shape)
            available_actions (np.ndarray): (num_agents, n_actions)
        Returns:
            actions (np.ndarray): sampled actions of agents
        """
        epsilon = np.random.random()
        if epsilon < self.exploration:
            available_actions = torch.tensor(available_actions,
                                             dtype=torch.float32)
            actions_dist = Categorical(available_actions)
            actions = actions_dist.sample().numpy()

        else:
            actions = self.predict(obs, available_actions)

        # update exploration
        self.exploration = max(self.ep_scheduler.step(), self.min_exploration)
        return actions

    def predict(self, obs: torch.Tensor, available_actions: torch.Tensor):
        """take greedy actions
        Args:
            obs (np.ndarray):               (num_agents, obs_shape)
            available_actions (np.ndarray): (num_agents, n_actions)
        Returns:
            actions (np.ndarray):           (num_agents, )
        """
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        available_actions = torch.tensor(available_actions,
                                         dtype=torch.long,
                                         device=self.device)
        policy_logits, self.hidden_state = self.actor_model(
            obs, self.hidden_state)
        # mask unavailable actions
        policy_logits[available_actions == 0] = -1e10
        actions = policy_logits.max(dim=1)[1].detach().cpu().numpy()
        return actions

    def update_target(self) -> None:
        hard_target_update(self.actor_model, self.target_actor_model)
        hard_target_update(self.critic_model, self.target_critic_model)

    def learn(self, episode_data: Dict[str, np.ndarray]):
        """Update the model from a batch of experiences.

        Args: episode_data (dict) with the following:

            - obs (np.ndarray):                     (batch_size, T, num_agents, obs_shape)
            - state (np.ndarray):                   (batch_size, T, state_shape)
            - actions (np.ndarray):                 (batch_size, T, num_agents)
            - rewards (np.ndarray):                  (batch_size, T, 1)
            - dones (np.ndarray):              (batch_size, T, 1)
            - available_actions (np.ndarray):        (batch_size, T, num_agents, n_actions)
            - filled (np.ndarray):                  (batch_size, T, 1)

        Returns:
            - mean_loss (float): train loss
            - mean_td_error (float): train TD error
        """
        # get the data from episode_data buffer
        obs_episode = episode_data['obs']
        state_episode = episode_data['state']
        actions_episode = episode_data['actions']
        available_actions_episode = episode_data['available_actions']
        rewards_episode = episode_data['rewards']
        dones_episode = episode_data['dones']
        filled_episode = episode_data['filled']
        # update target model
        if self.global_steps % self.update_target_interval == 0:
            self.update_target()
            self.target_update_count += 1

        self.global_steps += 1

        # set the actions to torch.Long
        actions_episode = actions_episode.to(self.device, dtype=torch.long)
        # get the batch_size and episode_length
        batch_size, episode_len = state_episode.shape[:2]

        # get the relevant quantitles
        rewards_episode = rewards_episode[:, :-1, :]
        actions_episode = actions_episode[:, :-1, :].unsqueeze(-1)
        dones_episode = dones_episode[:, :-1, :]
        filled_episode = filled_episode[:, :-1, :]

        mask = (1 - filled_episode) * (1 - dones_episode)

        # Calculate estimated Q-Values
        local_qs = []
        self.init_hidden_states(batch_size)
        for t in range(episode_len):
            obs = obs_episode[:, t, :, :]
            # obs: (batch_size * num_agents, obs_shape)
            obs = obs.reshape(-1, obs_episode.shape[-1])
            # Calculate estimated Q-Values
            local_q, self.hidden_state = self.actor_model(obs)
            #  local_q: (batch_size * num_agents, n_actions) -->  (batch_size, num_agents, n_actions)
            local_q = local_q.reshape(batch_size, self.num_agents, -1)
            local_qs.append(local_q)

        # Concat over time
        local_qs = torch.stack(local_qs, dim=1)
        local_qs[available_actions_episode == 0] = -1e10

        q_vals, critic_loss = self.learn_critic(obs_episode, actions_episode,
                                                rewards_episode, dones_episode,
                                                mask)
        # Calculate the baseline
        q_vals = q_vals.reshape(-1, self.n_actions)
        pi = local_qs.view(-1, self.n_actions)
        baseline = (pi * q_vals).sum(-1).detach()

        # Calculate policy grad with mask

        q_taken = torch.gather(q_vals,
                               dim=1,
                               index=actions_episode.reshape(-1, 1)).squeeze(1)
        pi_token = torch.gather(pi,
                                dim=1,
                                index=actions_episode.reshape(-1,
                                                              1)).squeeze(1)
        pi_token[mask == 0] = 1.0
        log_pi_taken = torch.log(pi_token)

        advantagees = (q_taken - baseline).detach()
        coma_loss = -((advantagees * log_pi_taken) * mask).sum() / mask.sum()

        dist_entropy = Categorical(pi).entropy().view(-1)
        dist_entropy[mask == 0] = 0
        entropy_loss = (dist_entropy * mask).sum() / mask.sum()

        # Optimise actor model

        self.actor_optimizer.zero_grad()
        actor_loss = coma_loss - self.entropy_coef * entropy_loss

        actor_loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.actor_params,
                                           self.clip_grad_norm)
        self.actor_optimizer.step()

        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.actor_lr

        return (actor_loss.item(), coma_loss.item(), critic_loss)

    def learn_critic(
        self,
        obs_episode: torch.Tensor,
        actions_episode: torch.Tensor,
        rewards_episode: torch.Tensor,
        dones_episode: torch.Tensor,
        mask_episode: torch.Tensor,
    ) -> None:
        """Update the critic model from a batch of experiences.

        Args: episode_data (dict) with the following:

            - obs (np.ndarray):                     (batch_size, T, num_agents, obs_shape)
            - state (np.ndarray):                   (batch_size, T, state_shape)
            - actions (np.ndarray):                 (batch_size, T, num_agents)
            - rewards (np.ndarray):                 (batch_size, T, 1)
            - dones (np.ndarray):                    (batch_size, T, 1)
            - available_actions (np.ndarray):        (batch_size, T, num_agents, n_actions)
            - filled (np.ndarray):                   (batch_size, T, 1)

        Returns:
            - critic_loss (float): train loss
        """
        # get the batch_size and episode_length
        batch_size, episode_len = obs_episode.shape[:2]

        # Optimise critic
        target_q_vals = self.target_critic_model(obs_episode)
        target_q_vals_taken = torch.gather(target_q_vals,
                                           dim=3,
                                           index=actions_episode).squeeze(3)
        # Calculate td-lambda targets
        targets = self.build_td_lambda_targets(
            rewards_episode,
            dones_episode,
            mask_episode,
            target_q_vals_taken,
            gamma=self.gamma,
            td_lambda=self.td_lambda,
        )
        q_vals = torch.zeros_like(target_q_vals)[:, :-1]

        critic_loss = []
        for t in reversed(episode_len):
            mask_t = mask_episode[:, t].expand(-1, self.num_agents)
            if mask_t.sum() == 0:
                continue

            q_t = self.critic_model()
            q_vals[:, t] = q_t.view(batch_size, self.num_agents,
                                    self.n_actions)
            q_taken = (torch.gather(
                q_t, dim=3,
                index=actions_episode[:, t:t + 1]).squeeze(3).squeeze(1))
            targets_t = targets[:, t]

            td_error = q_taken - targets_t.detach()

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask_t

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error**2).sum() / mask_t.sum()
            self.critic_optimizer.zero_grad()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.critic_params,
                                               self.clip_grad_norm)
            self.critic_optimiser.step()
            critic_loss.append(loss.item())

        return q_vals, np.mean(critic_loss)

    def build_td_lambda_targets(self, rewards, dones, mask, target_qs, gamma,
                                td_lambda):
        """Build TD-lambda targets for Q-learning.

        Args:
            rewards (torch.Tensor): rewards tensor, in shape (batch_size, T, 1)
            dones (torch.Tensor): dones tensor, in shape (batch_size, T, 1)
            mask (torch.Tensor): mask tensor, in shape (batch_size, T, 1)
            target_qs (torch.Tensor): target Q values tensor, in shape (batch_size, T, num_agents)
            gamma (float): discount factor
            td_lambda (float): lambda for TD-lambda return
        Returns:
            ret (torch.Tensor): lambda-return from t=0 to t=T-1, in shape (batch_size, T-1, num_agents)
        """
        # Initialise  last  lambda -return  for  not  terminated  episodes
        ret = target_qs.new_zeros(*target_qs.shape)
        ret[:, -1] = target_qs[:, -1] * (1 - torch.sum(dones, dim=1))
        # Backwards  recursive  update  of the "forward  view"
        for t in range(ret.shape[1] - 2, -1, -1):
            ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] * (
                rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] *
                (1 - dones[:, t]))
        # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
        return ret[:, 0:-1]

    def save(
        self,
        save_dir: str = None,
        actor_model_name: str = 'actor_model.th',
        critic_model_name: str = 'critic_model.th',
        opt_name: str = 'optimizer.th',
    ):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        actor_model_path = os.path.join(save_dir, actor_model_name)
        critic_model_path = os.path.join(save_dir, critic_model_name)
        optimizer_path = os.path.join(save_dir, opt_name)
        torch.save(self.actor_model.state_dict(), actor_model_path)
        torch.save(self.critic_model.state_dict(), critic_model_path)
        torch.save(self.critic_optimizer.state_dict(), optimizer_path)
        print('save model successfully!')

    def restore(
        self,
        save_dir: str = None,
        actor_model_name: str = 'actor_model.th',
        critic_model_name: str = 'critic_model.th',
        opt_name: str = 'optimizer.th',
    ):
        actor_model_path = os.path.join(save_dir, actor_model_name)
        critic_model_path = os.path.join(save_dir, critic_model_name)
        optimizer_path = os.path.join(save_dir, opt_name)
        self.actor_model.load_state_dict(torch.load(actor_model_path))
        self.critic_model.load_state_dict(torch.load(critic_model_path))
        self.critic_optimizer.load_state_dict(torch.load(optimizer_path))
        print('restore model successfully!')
