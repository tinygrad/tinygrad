from tinygrad import Tensor, TinyJit, nn
from tinygrad.helpers import dtypes  # TODO: wouldn't need this if argmax returned the right dtype
import gymnasium as gym
from tqdm import trange
import numpy as np  # TODO: remove numpy import

class Model:
  def __init__(self, in_features, out_features):
    self.l1 = nn.Linear(in_features, 32)
    self.l2 = nn.Linear(32, out_features)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.l1(x).tanh()
    return self.l2(x).log_softmax()

def evaluate(model:Model, test_env:gym.Env) -> float:
  (obs, _), done = test_env.reset(), False
  total_rew = 0.0
  while not done:
    act = model(Tensor(obs)).argmax().cast(dtypes.int32).item()
    obs, rew, done, _, _ = test_env.step(act)
    total_rew += rew
  return total_rew

# TODO: time should be < 5s on M1 Max
if __name__ == "__main__":
  env = gym.make('CartPole-v1')

  model = Model(env.observation_space.shape[0], int(env.action_space.n))
  opt = nn.optim.Adam(nn.state.get_parameters(model), lr=1e-2)

  @TinyJit
  def train_step(x:Tensor, y:Tensor) -> Tensor:
    opt.zero_grad()
    loss = (-(model(x) * y).sum(-1)).mean().backward()
    opt.step()
    return loss.realize()

  @TinyJit
  def get_action_dist(obs:Tensor) -> Tensor: return model(obs).exp().realize()

  BS = 128
  for i in (t:=trange(50)):
    X, Y = [], []
    ep_rews = []
    get_action_dist.reset()   # NOTE: if you don't reset the jit here it captures the wrong model on the first run through
    while len(X) < BS:
      obs:np.ndarray = env.reset()[0]
      acts, rews, done = [], [], False
      # NOTE: we don't want to early stop since then the rewards are wrong for the last episode
      while not done:
        # pick actions
        # TODO: move the multinomial into jitted tinygrad when JIT rand works
        # TODO: what's the temperature here?
        act = get_action_dist(Tensor(obs)).multinomial().item()

        # save this state action pair
        # TODO: don't use np.copy here on the CPU, what's the tinygrad way to do this and keep on device? need __setitem__ assignment
        X.append(np.copy(obs))
        acts.append(act)

        obs, rew, done, _, _ = env.step(act)
        rews.append(rew)
      ep_rews.append(sum(acts))

      # reward to go
      # TODO: move this into tinygrad
      for i, act in enumerate(acts):
        act_mask = np.zeros((env.action_space.n))
        act_mask[act] = np.sum(rews[i:])
        Y.append(act_mask)

    loss = train_step(Tensor(X[:BS]), Tensor(Y[:BS]))  # TODO: randomize this?
    t.set_description(f"loss: {loss.item():6.2f} ep_count: {len(ep_rews):2d} avg_ep_rew: {sum(ep_rews)/len(ep_rews):6.2f}")

  test_rew = evaluate(model, gym.make('CartPole-v1', render_mode='human'))
  print(f"test reward: {test_rew}")
