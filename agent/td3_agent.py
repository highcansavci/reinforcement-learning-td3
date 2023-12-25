import numpy as np
import torch
import torch.nn.functional as F
from replay_buffer.replay_buffer import ReplayBuffer
from model.actor import Actor
from model.critic import Critic


class Agent:
    def __init__(self, actor_lr, critic_lr, input_dims, tau, env, gamma=0.99, update_actor_interval=2, warmup=1000, n_actions=2, max_size=int(1e6), fc1_dim=400, fc2_dim=300, batch_size=100, noise=1e-1):
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.learn_step_cnt = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_interval = update_actor_interval
        self.noise = noise
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.actor = Actor(actor_lr, input_dims, fc1_dim, fc2_dim, n_actions, "actor")
        self.actor_target = Actor(actor_lr, input_dims, fc1_dim, fc2_dim, n_actions, "actor_target")
        self.critic_a = Critic(critic_lr, input_dims, fc1_dim, fc2_dim, n_actions, "critic_a")
        self.critic_b = Critic(critic_lr, input_dims, fc1_dim, fc2_dim, n_actions, "critic_b")
        self.critic_a_target = Critic(critic_lr, input_dims, fc1_dim, fc2_dim, n_actions, "critic_a_target")
        self.critic_b_target = Critic(critic_lr, input_dims, fc1_dim, fc2_dim, n_actions, "critic_b_target")
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = torch.tensor(np.random.normal(scale=self.noise, size=(self.n_actions, )), dtype=torch.float32, device=self.actor.device)
        else:
            state = torch.tensor(observation, dtype=torch.float32, device=self.actor.device)
            mu = self.actor(state).to(self.actor.device)
        mu = (mu - self.min_action) * (self.max_action - self.min_action) / 2 + self.min_action
        mu_prime = mu + torch.tensor(np.random.normal(scale=self.noise, size=(self.n_actions, )), dtype=torch.float32, device=self.actor.device)
        mu_prime = torch.clamp(mu_prime, self.min_action, self.max_action)
        self.time_step += 1
        return mu_prime.detach().cpu().numpy()

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.critic_a.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.critic_a.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.critic_a.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.critic_a.device)
        states = torch.tensor(states, dtype=torch.float32, device=self.critic_a.device)

        target_actions = self.actor_target(next_states)
        target_actions = (target_actions - self.min_action) * (self.max_action - self.min_action) / 2 + self.min_action
        target_actions = target_actions + torch.clamp(torch.tensor(np.random.normal(scale=0.2), dtype=torch.float32, device=self.critic_a.device), -0.5, 0.5)
        target_actions = torch.clamp(target_actions, self.min_action, self.max_action)
        q_a_target = self.critic_a_target(next_states, target_actions)
        q_b_target = self.critic_b_target(next_states, target_actions)
        q_a = self.critic_a(states, actions)
        q_b = self.critic_b(states, actions)
        q_a_target[dones] = 0.0
        q_b_target[dones] = 0.0

        critic_target_value = torch.min(q_a_target, q_b_target)
        target = rewards.reshape(self.batch_size, 1) + self.gamma * critic_target_value

        self.critic_a.optimizer.zero_grad()
        self.critic_b.optimizer.zero_grad()
        q_a_loss = F.mse_loss(target, q_a)
        q_b_loss = F.mse_loss(target, q_b)
        critic_loss = q_a_loss + q_b_loss
        critic_loss.backward()
        self.critic_a.optimizer.step()
        self.critic_b.optimizer.step()

        self.learn_step_cnt += 1
        if self.learn_step_cnt % self.update_actor_interval != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q_a_loss = -self.critic_a(states, self.actor(states)).mean()
        actor_q_a_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        self.soft_update(self.critic_a_target, self.critic_a, tau)
        self.soft_update(self.critic_b_target, self.critic_b, tau)
        self.soft_update(self.actor_target, self.actor, tau)

    def save_models(self):
        print("Saving Models...")
        self.actor.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic_a.save_checkpoint()
        self.critic_a_target.save_checkpoint()
        self.critic_b.save_checkpoint()
        self.critic_b_target.save_checkpoint()

    def load_models(self):
        print("Loading Models...")
        self.actor.load_checkpoint()
        self.actor_target.load_checkpoint()
        self.critic_a.load_checkpoint()
        self.critic_a_target.load_checkpoint()
        self.critic_b.load_checkpoint()
        self.critic_b_target.load_checkpoint()
