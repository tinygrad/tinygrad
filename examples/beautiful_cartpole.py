from typing import Tuple
import time
from tinygrad import Tensor, TinyJit, nn, Variable
from tinygrad.helpers import dtypes  # TODO: wouldn't need this if argmax returned the right dtype
import gymnasium as gym
from tqdm import trange
import numpy as np  # TODO: remove numpy import

class ActorCritic:
  def __init__(self, in_features, out_features, hidden_state=32):
    self.l1 = nn.Linear(in_features, hidden_state)
    self.l2 = nn.Linear(hidden_state, out_features)

    self.c1 = nn.Linear(in_features, hidden_state)
    self.c2 = nn.Linear(hidden_state, 1)

  def __call__(self, obs:Tensor) -> Tuple[Tensor, Tensor]:
    x = self.l1(obs).tanh()
    act = self.l2(x).log_softmax()
    x = self.c1(obs).relu()
    return act, self.c2(x)

def evaluate(model:ActorCritic, test_env:gym.Env) -> float:
  (obs, _), terminated, truncated = test_env.reset(), False, False
  total_rew = 0.0
  while not terminated and not truncated:
    act = model(Tensor(obs))[0].argmax().cast(dtypes.int32).item()
    obs, rew, terminated, truncated, _ = test_env.step(act)
    total_rew += float(rew)
  return total_rew

# TODO: time should be < 5s on M1 Max
if __name__ == "__main__":
  env = gym.make('CartPole-v1')

  model = ActorCritic(env.observation_space.shape[0], int(env.action_space.n))    # type: ignore
  opt = nn.optim.Adam(nn.state.get_parameters(model), lr=1e-2)

  @TinyJit
  def train_step(x:Tensor, selected_action:Tensor, reward:Tensor, old_log_dist:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    with Tensor.train():
      log_dist, value = model(x)

      # get advantage
      advantage = reward.reshape(-1, 1) - value
      mask = selected_action.reshape(-1, 1) == Tensor.arange(log_dist.shape[1]).reshape(1, -1).expand(selected_action.shape[0], -1)
      masked_advantage = mask * advantage.detach()

      # PPO
      ratios = (log_dist - old_log_dist).exp() * masked_advantage
      clipped_ratios = ratios.clip(1-0.2, 1+0.2) * masked_advantage
      action_loss = -ratios.minimum(clipped_ratios).sum(-1).mean()

      entropy_loss = (log_dist.exp() * log_dist).sum(-1).mean()   # this encourages diversity
      critic_loss = advantage.square().mean()
      opt.zero_grad()
      (action_loss + entropy_loss*0.0005 + critic_loss).backward()
      opt.step()
      return action_loss.realize(), entropy_loss.realize(), critic_loss.realize()

  @TinyJit
  def get_action_dist(obs:Tensor) -> Tensor:
    # TODO: with no_grad
    Tensor.no_grad = True
    ret = model(obs)[0].exp().realize()
    Tensor.no_grad = False
    return ret

  BS = 256
  MAX_REPLAY_BUFFER = 2000
  st, steps = time.perf_counter(), 0
  Xn, An, Rn = [], [], []
  for i in (t:=trange(40)):
    get_action_dist.reset()   # NOTE: if you don't reset the jit here it captures the wrong model on the first run through

    obs:np.ndarray = env.reset()[0]
    rews, terminated, truncated = [], False, False
    # NOTE: we don't want to early stop since then the rewards are wrong for the last episode
    while not terminated and not truncated:
      # pick actions
      # TODO: move the multinomial into jitted tinygrad when JIT rand works
      # TODO: what's the temperature here?
      act = get_action_dist(Tensor(obs)).multinomial().item()

      # save this state action pair
      # TODO: don't use np.copy here on the CPU, what's the tinygrad way to do this and keep on device? need __setitem__ assignment
      Xn.append(np.copy(obs))
      An.append(act)

      obs, rew, terminated, truncated, _ = env.step(act)
      rews.append(float(rew))
    steps += len(rews)

    # reward to go
    # TODO: move this into tinygrad
    discounts = np.power(0.99, np.arange(len(rews)))
    Rn += [np.sum(rews[i:] * discounts[:len(rews)-i]) for i in range(len(rews))]

    Xn, An, Rn = Xn[-MAX_REPLAY_BUFFER:], An[-MAX_REPLAY_BUFFER:], Rn[-MAX_REPLAY_BUFFER:]
    X, A, R = Tensor(Xn), Tensor(An), Tensor(Rn)

    # TODO: make this work
    #vsz = Variable("sz", 1, MAX_REPLAY_BUFFER-1).bind(len(Xn))
    #X, A, R = Tensor(Xn).reshape(vsz, None), Tensor(An).reshape(vsz), Tensor(Rn).reshape(vsz)

    old_log_dist = model(X)[0]   # TODO: could save these instead of recomputing
    for i in range(5):
      samples = Tensor.randint(BS, high=X.shape[0]).realize()  # TODO: remove the need for this
      # TODO: is this recompiling based on the shape?
      action_loss, entropy_loss, critic_loss = train_step(X[samples], A[samples], R[samples], old_log_dist[samples])
    t.set_description(f"sz: {len(Xn):5d} steps/s: {steps/(time.perf_counter()-st):7.2f} action_loss: {action_loss.item():7.2f} entropy_loss: {entropy_loss.item():7.2f} critic_loss: {critic_loss.item():7.2f} reward: {sum(rews):6.2f}")

  test_rew = evaluate(model, gym.make('CartPole-v1', render_mode='human'))
  print(f"test reward: {test_rew}")
