from typing import Optional, Tuple
from numpy.typing import NDArray

from tinygrad.tensor import Tensor
from tinygrad.nn import optim
from tinygrad.helpers import getenv

import numpy as np
import gym


DEVICE = "GPU" if getenv("GPU") else "CPU"


class Actor:
  def __init__(self, num_actions: int, num_states: int, hidden_size: Tuple[int, int] = (400, 300)):
    self.l1 = Tensor.glorot_uniform(num_states, hidden_size[0])
    self.l2 = Tensor.glorot_uniform(hidden_size[0], hidden_size[1])
    self.mu = Tensor.glorot_uniform(hidden_size[1], num_actions)

  def forward(self, state: Tensor, upper_bound: float) -> Tensor:
    out = state.dot(self.l1).relu()
    out = out.dot(self.l2).relu()
    out = out.dot(self.mu).tanh()
    output = out * upper_bound

    return output


class Critic:
  def __init__(self, num_inputs: int, hidden_size: Tuple[int, int] = (400, 300)):
    self.l1 = Tensor.glorot_uniform(num_inputs, hidden_size[0])
    self.l2 = Tensor.glorot_uniform(hidden_size[0], hidden_size[1])
    self.q = Tensor.glorot_uniform(hidden_size[1], 1)

  def forward(self, state: Tensor, action: Tensor) -> Tensor:
    inputs = state.cat(action, dim=1)
    out = inputs.dot(self.l1).relu()
    out = out.dot(self.l2).relu()
    q = out.dot(self.q)

    return q


class Buffer:
  def __init__(self, num_actions: int, num_states: int, buffer_capacity: int = 100000, batch_size: int = 64):
    self.buffer_capacity = buffer_capacity
    self.batch_size = batch_size

    self.buffer_counter = 0

    self.state_buffer = np.zeros((self.buffer_capacity, num_states))
    self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
    self.reward_buffer = np.zeros((self.buffer_capacity, 1))
    self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))
    self.done_buffer = np.zeros((self.buffer_capacity, 1))

  def record(
    self, observations: Tuple[Tensor, NDArray, float, NDArray, bool]
  ) -> None:
    index = self.buffer_counter % self.buffer_capacity

    self.state_buffer[index] = observations[0].detach().numpy()
    self.action_buffer[index] = observations[1]
    self.reward_buffer[index] = observations[2]
    self.next_state_buffer[index] = observations[3]
    self.done_buffer[index] = observations[4]

    self.buffer_counter += 1

  def sample(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    record_range = min(self.buffer_counter, self.buffer_capacity)
    batch_indices = np.random.choice(record_range, self.batch_size)

    state_batch = Tensor(self.state_buffer[batch_indices], device=DEVICE, requires_grad=False)
    action_batch = Tensor(self.action_buffer[batch_indices], device=DEVICE, requires_grad=False)
    reward_batch = Tensor(self.reward_buffer[batch_indices], device=DEVICE, requires_grad=False)
    next_state_batch = Tensor(self.next_state_buffer[batch_indices], device=DEVICE, requires_grad=False)
    done_batch = Tensor(self.done_buffer[batch_indices], device=DEVICE, requires_grad=False)

    return state_batch, action_batch, reward_batch, next_state_batch, done_batch


class GaussianActionNoise:
  def __init__(self, mean: NDArray, std_deviation: NDArray):
    self.mean = mean
    self.std_dev = std_deviation

  def __call__(self) -> Tensor:
    return Tensor(
        np.random.default_rng()
        .normal(self.mean, self.std_dev, size=self.mean.shape)
        .astype(np.float32),
        device=DEVICE,
        requires_grad=False,
    )


class DeepDeterministicPolicyGradient:
  """Deep Deterministic Policy Gradient (DDPG).

  https://arxiv.org/pdf/1509.02971.pdf

  Args:
      env: The environment to learn from.
      lr_actor: The learning rate of the actor.
      lr_critic: The learning rate of the critic.
      gamma: The discount factor.
      buffer_capacity: The size of the replay buffer.
      tau: The soft update coefficient.
      hidden_size: The number of neurons in the hidden layers of the actor and critic networks.
      batch_size: The minibatch size for each gradient update.
      noise_stddev: The standard deviation of the exploration noise.

  Note:
      In contrast to the original paper, actions are already included in the first layer 
      of the Critic and we use a Gaussian distribution instead of an Ornstein Uhlenbeck 
      process for exploration noise.

  """

  def __init__(
    self,
    env: gym.Env,
    lr_actor: float = 0.001,
    lr_critic: float = 0.002,
    gamma: float = 0.99,
    buffer_capacity: int = 100000,
    tau: float = 0.005,
    hidden_size: Tuple[int, int] = (400, 300),
    batch_size: int = 64,
    noise_stddev: float = 0.1,
  ):
    self.num_states =  env.observation_space.shape[0]
    self.num_actions =  env.action_space.shape[0]
    self.max_action =  env.action_space.high.item()
    self.min_action =  env.action_space.low.item()
    self.gamma = gamma
    self.tau = tau
    self.memory = Buffer(
        self.num_actions, self.num_states, buffer_capacity, batch_size
    )
    self.batch_size = batch_size

    self.noise = GaussianActionNoise(
        mean=np.zeros(self.num_actions),
        std_deviation=noise_stddev * np.ones(self.num_actions),
    )

    self.actor = Actor(self.num_actions, self.num_states, hidden_size)
    self.critic = Critic(self.num_actions + self.num_states, hidden_size)
    self.target_actor = Actor(self.num_actions, self.num_states, hidden_size)
    self.target_critic = Critic(self.num_actions + self.num_states, hidden_size)

    actor_params = optim.get_parameters(self.actor)
    critic_params = optim.get_parameters(self.critic)
    target_actor_params = optim.get_parameters(self.target_actor)
    target_critic_params = optim.get_parameters(self.target_critic)

    if DEVICE == "GPU":
      [x.gpu_() for x in actor_params + critic_params + target_actor_params + target_critic_params]

    self.actor_optimizer = optim.Adam(actor_params, lr_actor)
    self.critic_optimizer = optim.Adam(critic_params, lr_critic)

    self.update_network_parameters(tau=1.0)

  def update_network_parameters(self, tau: Optional[float] = None) -> None:
    """Updates the parameters of the target networks via 'soft updates'."""
    if tau is None:
      tau = self.tau

    for param, target_param in zip(
        optim.get_parameters(self.actor), optim.get_parameters(self.target_actor)
    ):
      target_param.assign(param * tau + target_param * (1.0 - tau))

    for param, target_param in zip(
        optim.get_parameters(self.critic), optim.get_parameters(self.target_critic)
    ):
      target_param.assign(param * tau + target_param * (1.0 - tau))

  def choose_action(self, state: Tensor, evaluate: bool = False) -> NDArray:
    mu = self.actor.forward(state, self.max_action)

    if not evaluate:
      mu = mu.add(self.noise())

    mu = mu.clip(self.min_action, self.max_action)

    return mu.detach().numpy()

  def learn(self) -> None:
    """Performs a learning step by sampling from replay buffer and updating networks."""
    if self.memory.buffer_counter < self.batch_size:
      return

    (
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        done_batch,
    ) = self.memory.sample()
 
    target_actions = self.target_actor.forward(next_state_batch, self.max_action)
    y = reward_batch + self.gamma * self.target_critic.forward(
        next_state_batch, target_actions.detach()
    ) * (Tensor.ones(*done_batch.shape, device=DEVICE, requires_grad=False) - done_batch)

    self.critic_optimizer.zero_grad()
    critic_value = self.critic.forward(state_batch, action_batch)
    critic_loss = y.detach().sub(critic_value).pow(2).mean()
    critic_loss.backward()
    self.critic_optimizer.step()

    self.actor_optimizer.zero_grad()
    actions = self.actor.forward(state_batch, self.max_action)
    critic_value = self.critic.forward(state_batch, actions)
    actor_loss = -critic_value.mean()
    actor_loss.backward()
    self.actor_optimizer.step()

    self.update_network_parameters()


if __name__ == "__main__":
  env = gym.make("Pendulum-v1")
  agent = DeepDeterministicPolicyGradient(env)
  num_episodes = 150

  for episode in range(1, num_episodes+1):
    cumulative_reward = 0.0
    prev_state, info = env.reset()  # for older gym versions only state is returned, so remove info
    done = False

    while not done:
      prev_state = Tensor(prev_state, device=DEVICE, requires_grad=False)
      action = agent.choose_action(prev_state)

      state, reward, done, _, info = env.step(action)  # for older gym versions there is only one bool, so remove _

      cumulative_reward += reward

      agent.memory.record((prev_state, action, reward, state, done))
      agent.learn()

      if done:
        break

      prev_state = state

    print(
        f"Episode {episode}/{num_episodes} - cumulative reward: {cumulative_reward}"
    )
