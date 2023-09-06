from typing import Optional, Tuple
from numpy.typing import NDArray

from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor
from tinygrad.nn import optim
from tinygrad.helpers import getenv

import numpy as np
import gymnasium as gym

"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
Continuous version by Ian Danforth
"""

import math
from gymnasium import spaces, logger
from gymnasium.utils import seeding
import numpy as np

#source: https://gist.github.com/iandanforth/e3ffb67cf3623153e968f2afdfb01dc8
class ContinuousCartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, render_mode = None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0

        if render_mode in self.__class__.metadata['render.modes']+[None]:
          self.render_mode = render_mode
        else:
          raise( " render_mode <{render_mode}> not in ", self.__class__.metadata['render.modes']+[None])
        
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])
        
        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force):
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
            (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return (x, x_dot, theta, theta_dot)

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        # Cast action to float to strip np trappings
        force = self.force_mag * action[0]
        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot = self.state
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("""
You are calling 'step()' even though this environment has already returned
done = True. You should always call 'reset()' once you receive 'done = True'
Any further steps are undefined behavior.
                """)
            self.steps_beyond_done += 1
            reward = 0.0
        
        if self.render_mode == "rgb_array":
            self.state = self.render(self, mode=self.render_mode)
        if self.render_mode == "human":
            self.render(self, mode=self.render_mode)
            
        return np.array(self.state), reward, done, {}, None

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state), None

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width /world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gymnasium.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen-polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array')), None

    def close(self):
        if self.viewer:
            self.viewer.close()

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

    self.state_buffer = np.zeros((self.buffer_capacity, num_states), np.float32)
    self.action_buffer = np.zeros((self.buffer_capacity, num_actions), np.float32)
    self.reward_buffer = np.zeros((self.buffer_capacity, 1), np.float32)
    self.next_state_buffer = np.zeros((self.buffer_capacity, num_states), np.float32)
    self.done_buffer = np.zeros((self.buffer_capacity, 1), np.float32)

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

    state_batch = Tensor(self.state_buffer[batch_indices], requires_grad=False)
    action_batch = Tensor(self.action_buffer[batch_indices], requires_grad=False)
    reward_batch = Tensor(self.reward_buffer[batch_indices], requires_grad=False)
    next_state_batch = Tensor(self.next_state_buffer[batch_indices], requires_grad=False)
    done_batch = Tensor(self.done_buffer[batch_indices], requires_grad=False)

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
    lr_actor: float = 0.00001,
    lr_critic: float = 0.0001,
    gamma: float = 0.99,
    buffer_capacity: int = 500000,
    tau: float = 0.001,
    hidden_size: Tuple[int, int] = (400, 300),
    batch_size: int = 32,
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

    actor_params = get_parameters(self.actor)
    critic_params = get_parameters(self.critic)
    target_actor_params = get_parameters(self.target_actor)
    target_critic_params = get_parameters(self.target_critic)

    self.actor_optimizer = optim.AdamW(actor_params, lr_actor,wd = 0.001)
    self.critic_optimizer = optim.AdamW(critic_params, lr_critic,wd = 0.001)

    self.update_network_parameters(tau=1.0)

  def update_network_parameters(self, tau: Optional[float] = None) -> None:
    """Updates the parameters of the target networks via 'soft updates'."""
    if tau is None:
      tau = self.tau

    for param, target_param in zip(
        get_parameters(self.actor), get_parameters(self.target_actor)
    ):
      target_param.assign(param.detach() * tau + target_param * (1.0 - tau))

    for param, target_param in zip(
        get_parameters(self.critic), get_parameters(self.target_critic)
    ):
      target_param.assign(param.detach() * tau + target_param * (1.0 - tau))

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
    ) * (Tensor.ones(*done_batch.shape, requires_grad=False) - done_batch)

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
    print("LOSS: ",actor_loss.numpy(),critic_loss.numpy())

    self.update_network_parameters()


if __name__ == "__main__":
  
  #check for gymnasium
  import subprocess
  import sys
  import os

  def install_gymnasium():
    subprocess.run(["pip", "install", "gymnasium[all]", "gymnasium[atari]", "gymnasium[accept-rom-license]"])
    # to uninstall :
    #   "pip uninstall gymnasium[all] gymnasium[atari] gymnasium[accept-rom-license]"
    #   subprocess.run(["pip", "uninstall", "gymnasium[all]", "gymnasium[atari]", "gymnasium[accept-rom-license]"])

  try: import gymnasium as gym
  except ImportError:
    install_choice = input("Gymnasium is not installed. Install now? (y/n): ")
    if install_choice.lower() == ['y','ye''yes']:
      install_gymnasium()
      print("Gymnasium installed. Restarting the script.")
      os.execv(sys.executable, ['python'] + sys.argv)
    else: print("Gymnasium is required. Exiting."), sys.exit(1)
    
  #base on https://ijarsct.co.in/Paper943.pdf [Page:9]
  #Hyperparameter:
  #   env: The environment to learn from.
  # lr_actor: The learning rate of the actor.
  # lr_critic: The learning rate of the critic.
  # gamma: The discount factor.
  # buffer_capacity: The size of the replay buffer.
  # tau: The soft update coefficient.
  # hidden_size: The number of neurons in the hidden layers of the actor and critic networks.
  # batch_size: The minibatch size for each gradient update.
  # noise_stddev: The standard deviation of the exploration noise.
  
  #Hyperparameter:
  num_episodes = 20000
  max_step = 200
  number_of_simulation_before_training = 10
  number_of_training_before_simulation = 10
  print_interval = 10
  
  lr_actor = 0.00001
  lr_critic = 0.001
  gamma= 0.99
  buffer_capacity = 500
  tau = 0.005
  hidden_size = (400, 300)
  batch_size = 128
  noise_stddev = 0.2
  
  np.random.seed(137)#42
  
  #init parameter
  cumulative_reward = 0.
  max_reward=[0.]
  # env = gym.make("Pendulum-v1", render_mode="human", g=9.81)
  env = ContinuousCartPoleEnv()
  if spaces.Box != env.action_space.__class__:
    raise("The environment need to have a continuous action space")
  
  agent = DeepDeterministicPolicyGradient(    
          env= env,
          lr_actor=lr_actor,
          lr_critic=lr_critic,
          gamma=gamma,
          buffer_capacity = buffer_capacity,
          tau=tau,
          hidden_size=hidden_size,
          batch_size=batch_size,
          noise_stddev=noise_stddev
                                          )

  for episode in range(1, num_episodes+1):
    prev_state, info = env.reset()  
    done = False
    step_count = 0 
    while not done or max_step == step_count:
      prev_state = Tensor(prev_state, requires_grad=False)
      action = agent.choose_action(prev_state)
      state, reward, done, _, info = env.step(action)  
      cumulative_reward += reward
      agent.memory.record((prev_state, action, reward, state, done))
      prev_state = state
      step_count+=1
    
    if number_of_simulation_before_training == 0:
      for i in range(number_of_training_before_simulation):
        agent.lean()
      # [ agent.learn() for i in range(number_of_training_before_simulation)]


    if episode%print_interval==0:
      avg_reward = cumulative_reward/print_interval
      print(f"Episode {episode}/{num_episodes} - average reward: {cumulative_reward/print_interval} of the last {print_interval} episode")
      cumulative_reward = 0.
      # # save best model
      # if avg_reward >= max(max_reward): 
      #   print("ONE @!!")
      #   max_reward.append(avg_reward)
        # from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
        # agent_checkpoint = get_state_dict(agent)
        # safe_save(agent_checkpoint, "ddpg.safetensors")
      
  ### Play agent
  # from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
  # # # agent_checkpoint = get_state_dict(agent)
  # # # safe_save(agent_checkpoint, "ddpg.safetensors")
  # agent = safe_load("ddpg.safetensors")
  # load_state_dict(DeepDeterministicPolicyGradient, agent_checkpoint)
  # # env = gym.make("Pendulum-v1", render_mode="human", g=9.81)
  # env = ContinuousCartPoleEnv(render_mode="human")
  # if spaces.Box != env.action_space.__class__:
  #   raise("The environment need to have a continuous action space")
  # prev_state, info = env.reset()  
  # done = False
  # step_count = 0 
  # import time
  # number_of_frame_per_second=24
  # while not done:
  #   prev_state = Tensor(prev_state, requires_grad=False)
  #   action = agent.choose_action(prev_state)
  #   state, reward, done, _, info = env.step(action)  
  #   cumulative_reward += reward
  #   agent.memory.record((prev_state, action, reward, state, done))
  #   prev_state = state
  #   step_count+=1
  #   time.sleep(number_of_frame_per_second)
  #   if max_step == step_count:
  #     break
  # env.close()
