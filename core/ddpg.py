import copy
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def init_linear(module, output=False):
    if output:
        nn.init.uniform_(module.weight, -3.0e-3, 3.0e-3)
    else:
        nn.init.kaiming_uniform_(module.weight, a=0.01)
    nn.init.zeros_(module.bias)


class ResidualConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, spatial_side):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.residual = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.norm1 = nn.LayerNorm((output_channels, spatial_side, spatial_side))
        self.norm2 = nn.LayerNorm((output_channels, spatial_side, spatial_side))
        for layer in (self.conv1, self.conv2, self.residual):
            nn.init.kaiming_uniform_(layer.weight, a=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, value):
        residual = self.residual(value)
        value = F.leaky_relu(self.norm1(self.conv1(value)), negative_slope=0.01)
        value = self.norm2(self.conv2(value))
        return F.leaky_relu(value + residual, negative_slope=0.01)


@dataclass
class DDPGConfig:
    convolution_channels: int = 64
    actor_fc_width: int = 512
    critic_hidden_1: int = 1024
    critic_hidden_2: int = 512
    actor_lr: float = 4.0e-4
    critic_lr: float = 4.0e-4
    gamma: float = 0.99
    tau: float = 0.01
    batch_size: int = 32
    replay_capacity: int = 5_000
    learning_rate_patience: int = 200
    learning_rate_decay: float = 0.8
    minimum_learning_rate: float = 1.0e-6
    noise_clip: float = 2.0
    noise_variance_start: float = 2.0
    noise_decay: float = 0.95
    noise_gap: float = 100.0


def next_multiple(value, multiple=64):
    return (value // multiple + 1) * multiple


def make_config(L, N, K):
    action_dim = 2 * L * N + K
    state_dim = 1 + action_dim + 2 * K * N
    power_branch_input = 1 + K + 2 * K * N
    return DDPGConfig(
        actor_fc_width=max(512, next_multiple(power_branch_input)),
        critic_hidden_1=max(1024, next_multiple(max(state_dim, action_dim))),
        critic_hidden_2=max(512, next_multiple(action_dim)),
    )


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, seed, device):
        self.capacity = int(capacity)
        self.device = device
        self.states = torch.empty((capacity, state_dim), device=device, dtype=torch.float32)
        self.actions = torch.empty((capacity, action_dim), device=device, dtype=torch.float32)
        self.rewards = torch.empty((capacity, 1), device=device, dtype=torch.float32)
        self.next_states = torch.empty((capacity, state_dim), device=device, dtype=torch.float32)
        self.dones = torch.empty((capacity, 1), device=device, dtype=torch.float32)
        self.position = 0
        self.size = 0
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def add_batch(self, state, action, reward, next_state, done):
        tensors = (
            state.detach(),
            action.detach(),
            reward.detach().reshape(-1, 1),
            next_state.detach(),
            done.detach().reshape(-1, 1),
        )
        batch_size = int(tensors[0].shape[0])
        if batch_size >= self.capacity:
            tensors = tuple(value[-self.capacity:] for value in tensors)
            batch_size = self.capacity
        indices = (torch.arange(batch_size, device=self.device) + self.position) % self.capacity
        for destination, value in zip(
            (self.states, self.actions, self.rewards, self.next_states, self.dones),
            tensors,
        ):
            destination.index_copy_(0, indices, value)
        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        indices = torch.randint(
            self.size,
            (batch_size,),
            device=self.device,
            generator=self.generator,
        )
        return tuple(
            value.index_select(0, indices)
            for value in (self.states, self.actions, self.rewards, self.next_states, self.dones)
        )


class Actor(nn.Module):
    def __init__(self, L, N, K, action_dim, convolution_channels, fc_width):
        super().__init__()
        side = math.isqrt(N)
        self.layers = L
        self.elements = N
        self.users = K
        self.action_dim = action_dim
        self.prefix_dim = 1
        self.side = side
        self.phase_size = L * N
        self.phase_control_dim = 2 * self.phase_size
        spatial_channels = 2 * L
        self.block1 = ResidualConvBlock(spatial_channels, convolution_channels, side)
        self.block2 = ResidualConvBlock(convolution_channels, convolution_channels, side)
        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.phase_dense = nn.Linear(convolution_channels * 3 * 3, self.phase_control_dim)
        power_input = self.prefix_dim + K + 2 * K * N
        self.power_fc1 = nn.Linear(power_input, fc_width)
        self.power_fc2 = nn.Linear(fc_width, fc_width)
        self.power_fc3 = nn.Linear(fc_width, K)
        init_linear(self.phase_dense, output=True)
        init_linear(self.power_fc1)
        init_linear(self.power_fc2)
        init_linear(self.power_fc3, output=True)

    def forward(self, state):
        auxiliary = state[:, :self.prefix_dim]
        previous_action = state[:, self.prefix_dim:self.prefix_dim + self.action_dim]
        csi = state[:, self.prefix_dim + self.action_dim:]
        previous_real = previous_action[:, :self.phase_size].reshape(
            -1, self.layers, self.side, self.side,
        )
        previous_imag = previous_action[:, self.phase_size:2 * self.phase_size].reshape(
            -1, self.layers, self.side, self.side,
        )
        previous_power = previous_action[:, 2 * self.phase_size:]
        phase_spatial = torch.cat((previous_real, previous_imag), dim=1)
        phase_features = self.pool(self.block2(self.block1(phase_spatial))).flatten(1)
        phase_output = torch.tanh(self.phase_dense(phase_features))

        power_features = torch.cat((auxiliary, previous_power, csi), dim=1)
        power_features = F.leaky_relu(self.power_fc1(power_features), negative_slope=0.01)
        power_features = F.leaky_relu(self.power_fc2(power_features), negative_slope=0.01)
        power_output = torch.softmax(self.power_fc3(power_features), dim=1)
        return torch.cat((phase_output, power_output), dim=1)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_1, hidden_2):
        super().__init__()
        self.state_branch = nn.Linear(state_dim, hidden_1)
        self.action_branch = nn.Linear(action_dim, hidden_1)
        self.fc1 = nn.Linear(hidden_1, hidden_1)
        self.norm1 = nn.LayerNorm(hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.norm2 = nn.LayerNorm(hidden_2)
        self.fc3 = nn.Linear(hidden_2, 1)
        init_linear(self.state_branch)
        init_linear(self.action_branch)
        init_linear(self.fc1)
        init_linear(self.fc2)
        init_linear(self.fc3, output=True)

    def forward(self, state, action):
        value = F.leaky_relu(
            self.state_branch(state) + self.action_branch(action),
            negative_slope=0.01,
        )
        value = F.leaky_relu(self.norm1(self.fc1(value)), negative_slope=0.01)
        value = F.leaky_relu(self.norm2(self.fc2(value)), negative_slope=0.01)
        return self.fc3(value)


class DDPGEnvironment:
    def __init__(self, W_T_1, W_T, pathloss, L, N, K, Pt, sigma2, qos_threshold, device):
        self.layers = int(L)
        self.elements = int(N)
        self.users = int(K)
        self.total_power = float(Pt)
        self.noise_power = float(sigma2)
        self.qos_threshold = float(qos_threshold)
        self.device = device
        self.phase_size = self.layers * self.elements
        self.phase_control_dim = 2 * self.phase_size
        self.action_dim = self.phase_control_dim + self.users
        self.csi_dim = 2 * self.users * self.elements
        self.state_dim = 1 + self.action_dim + self.csi_dim

        self.W_T_1 = torch.as_tensor(W_T_1, device=device).cfloat()
        self.W_T = torch.as_tensor(W_T, device=device).cfloat()
        pathloss_tensor = torch.as_tensor(pathloss, device=device, dtype=torch.float32)
        self.pathloss_sqrt = torch.sqrt(pathloss_tensor).view(1, K, 1)

    def encode_csi(self, channel):
        normalized = channel / self.pathloss_sqrt
        return torch.cat(
            (
                normalized.real.reshape(channel.shape[0], -1),
                normalized.imag.reshape(channel.shape[0], -1),
            ),
            dim=1,
        ).float()

    def initial_action(self, count, generator):
        theta = 2.0 * math.pi * torch.rand(
            (count, self.phase_size),
            device=self.device,
            generator=generator,
        )
        power_controls = torch.full((count, self.users), 1.0 / self.users, device=self.device)
        return torch.cat((torch.cos(theta), torch.sin(theta), power_controls), dim=1)

    def canonicalize_action(self, action):
        real = action[:, :self.phase_size]
        imag = action[:, self.phase_size:2 * self.phase_size]
        norm = torch.sqrt(real.square() + imag.square())
        safe_norm = norm.clamp_min(1.0e-8)
        normalized_real = real / safe_norm
        normalized_imag = imag / safe_norm
        zero_pair = norm < 1.0e-8
        normalized_real = torch.where(zero_pair, torch.ones_like(normalized_real), normalized_real)
        normalized_imag = torch.where(zero_pair, torch.zeros_like(normalized_imag), normalized_imag)
        raw_power = action[:, self.phase_control_dim:].clamp_min(0.0)
        power_sum = raw_power.sum(dim=1, keepdim=True)
        uniform_power = torch.full_like(raw_power, 1.0 / self.users)
        power_controls = torch.where(
            power_sum > 1.0e-8,
            raw_power / power_sum.clamp_min(1.0e-8),
            uniform_power,
        )
        return torch.cat((normalized_real, normalized_imag, power_controls), dim=1)

    def build_state(self, channel, previous_action, previous_reward):
        return torch.cat(
            (previous_reward.reshape(-1, 1), previous_action, self.encode_csi(channel)),
            dim=1,
        )

    def physical_metrics(self, channel, action):
        canonical = self.canonicalize_action(action)
        real = canonical[:, :self.phase_size]
        imag = canonical[:, self.phase_size:2 * self.phase_size]
        phase = torch.complex(real, imag).reshape(-1, self.layers, self.elements)
        power = self.total_power * canonical[:, self.phase_control_dim:]
        W_sim = phase[:, 0].unsqueeze(-1) * self.W_T_1.unsqueeze(0)
        W_T_batch = self.W_T.unsqueeze(0)
        for layer in range(1, self.layers):
            W_sim = phase[:, layer].unsqueeze(-1) * torch.matmul(W_T_batch, W_sim)
        effective_channel = torch.matmul(channel, W_sim)
        gains = torch.abs(effective_channel).square()
        desired = torch.diagonal(gains, dim1=1, dim2=2)
        signal = power * desired
        total_received = torch.sum(gains * power.unsqueeze(1), dim=2)
        interference = total_received - signal
        rates = torch.log2(1.0 + signal / (interference + self.noise_power))
        return {
            "sum_rate": rates.sum(dim=1),
            "rates": rates,
            "minimum_rate": rates.min(dim=1).values,
            "feasible": (rates >= self.qos_threshold - 1.0e-7).all(dim=1),
        }


def make_actor(environment, config):
    return Actor(
        environment.layers,
        environment.elements,
        environment.users,
        environment.action_dim,
        config.convolution_channels,
        config.actor_fc_width,
    ).to(environment.device)


@torch.no_grad()
def evaluate_actor(environment, actor, channels, rollout_steps, restarts, seed):
    device = environment.device
    channels = channels.to(device)
    count = channels.shape[0]
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    actor.eval()
    best_sum_rate = torch.full((count,), -math.inf, device=device)
    best_minimum_rate = torch.full((count,), -math.inf, device=device)
    best_feasible = torch.zeros(count, dtype=torch.bool, device=device)

    for _ in range(restarts):
        previous_action = environment.initial_action(count, generator)
        previous_reward = torch.zeros(count, device=device)
        for step in range(rollout_steps + 1):
            if step:
                state = environment.build_state(channels, previous_action, previous_reward)
                previous_action = environment.canonicalize_action(actor(state))
            metrics = environment.physical_metrics(channels, previous_action)
            previous_reward = metrics["sum_rate"]
            better = metrics["sum_rate"] > best_sum_rate
            best_feasible = torch.where(better, metrics["feasible"], best_feasible)
            best_sum_rate = torch.where(better, metrics["sum_rate"], best_sum_rate)
            best_minimum_rate = torch.where(better, metrics["minimum_rate"], best_minimum_rate)

    return (
        best_sum_rate.cpu().numpy(),
        best_minimum_rate.cpu().numpy(),
        best_feasible.cpu().numpy(),
    )


class DDPGTrainer:
    def __init__(self, environment, config, seed):
        self.environment = environment
        self.config = config
        self.device = environment.device
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.actor = make_actor(environment, config)
        self.critic = Critic(
            environment.state_dim,
            environment.action_dim,
            config.critic_hidden_1,
            config.critic_hidden_2,
        ).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).eval()
        self.target_critic = copy.deepcopy(self.critic).eval()
        for network in (self.target_actor, self.target_critic):
            for parameter in network.parameters():
                parameter.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.replay = None

    @staticmethod
    def soft_update(target, source, tau):
        with torch.no_grad():
            for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
                target_parameter.lerp_(source_parameter, tau)

    def decay_learning_rates(self):
        cfg = self.config
        for optimizer in (self.actor_optimizer, self.critic_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = max(
                    cfg.minimum_learning_rate,
                    float(group["lr"]) * cfg.learning_rate_decay,
                )

    def update(self):
        cfg = self.config
        state, action, reward, next_state, done = self.replay.sample(cfg.batch_size)
        with torch.no_grad():
            target_action = self.environment.canonicalize_action(self.target_actor(next_state))
            target_q = self.target_critic(next_state, target_action)
            target = reward + cfg.gamma * (1.0 - done) * target_q
        predicted = self.critic(state, action)
        critic_loss = F.mse_loss(predicted, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_action = self.environment.canonicalize_action(self.actor(state))
        actor_loss = -self.critic(state, actor_action).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.soft_update(self.target_actor, self.actor, cfg.tau)
        self.soft_update(self.target_critic, self.critic, cfg.tau)

    def train_episode(self, channels, steps_per_episode, seed, progress_interval=5_000):
        cfg = self.config
        environment = self.environment
        channels = channels.to(self.device)
        count = channels.shape[0]
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed + 101)
        self.replay = ReplayBuffer(
            cfg.replay_capacity,
            environment.state_dim,
            environment.action_dim,
            seed=seed + 1,
            device=self.device,
        )
        stale_steps = 0
        best_recent_reward = -math.inf
        started = time.perf_counter()

        previous_action = environment.initial_action(count, generator)
        with torch.no_grad():
            metrics = environment.physical_metrics(channels, previous_action)
        state = environment.build_state(channels, previous_action, metrics["sum_rate"])
        best_sum = metrics["sum_rate"].clone()
        best_action = previous_action.clone()

        self.actor.train()
        for step in range(steps_per_episode):
            with torch.no_grad():
                raw_action = self.actor(state)
                variance = cfg.noise_variance_start * (cfg.noise_decay ** (step / cfg.noise_gap))
                noise = math.sqrt(max(variance, 0.0)) * torch.randn(
                    raw_action.shape,
                    device=self.device,
                    generator=generator,
                )
                noise = noise.clamp(-cfg.noise_clip, cfg.noise_clip)
                action = environment.canonicalize_action(raw_action + noise)
                metrics = environment.physical_metrics(channels, action)
                reward = metrics["sum_rate"]
                done = torch.full(
                    (count,),
                    float(step + 1 == steps_per_episode),
                    device=self.device,
                )
                next_state = environment.build_state(channels, action, reward)
                better = reward > best_sum
                best_sum = torch.where(better, reward, best_sum)
                best_action = torch.where(better.unsqueeze(1), action, best_action)

            self.replay.add_batch(state, action, reward, next_state, done)
            state = next_state
            if self.replay.size >= max(cfg.replay_capacity, cfg.batch_size):
                self.update()

            current_reward = float(reward.mean().cpu())
            if current_reward > best_recent_reward + 1.0e-5:
                best_recent_reward = current_reward
                stale_steps = 0
            else:
                stale_steps += 1
            if stale_steps >= cfg.learning_rate_patience:
                self.decay_learning_rates()
                stale_steps = 0
                best_recent_reward = current_reward

            if progress_interval > 0 and (step + 1) % progress_interval == 0:
                print(
                    f"step={step + 1}/{steps_per_episode} "
                    f"explore-sum={float(reward.mean().cpu()):.3f} "
                    f"best-sum={float(best_sum.mean().cpu()):.3f} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )

        return best_sum, best_action
