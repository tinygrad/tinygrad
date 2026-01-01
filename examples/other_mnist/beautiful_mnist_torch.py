from tinygrad import dtypes, getenv, Device, Tensor, TinyJit
from tinygrad.helpers import trange, colored, DEBUG, temp
from tinygrad.nn.datasets import mnist
import torch
from torch import nn, optim

from extra.torch_backend.backend import unwrap, wrap
from torch._dynamo.backends.registry import register_backend
from torch._functorch.aot_autograd import aot_module_simplified

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.c1 = nn.Conv2d(1, 32, 5)
    self.c2 = nn.Conv2d(32, 32, 5)
    self.bn1 = nn.BatchNorm2d(32)
    self.m1 = nn.MaxPool2d(2)
    self.c3 = nn.Conv2d(32, 64, 3)
    self.c4 = nn.Conv2d(64, 64, 3)
    self.bn2 = nn.BatchNorm2d(64)
    self.m2 = nn.MaxPool2d(2)
    self.lin = nn.Linear(576, 10)
  def forward(self, x):
    x = nn.functional.relu(self.c1(x))
    x = nn.functional.relu(self.c2(x), 0)
    x = self.m1(self.bn1(x))
    x = nn.functional.relu(self.c3(x), 0)
    x = nn.functional.relu(self.c4(x), 0)
    x = self.m2(self.bn2(x))
    return self.lin(torch.flatten(x, 1))


@register_backend
def tiny(gm:torch.fx.GraphModule, sample_inputs):
  def my_compiler(gm:torch.fx.GraphModule, sample_inputs):
    # TODO: the jit should capture the graph directly, not need three runs. this is a planned tinygrad refactor after becomes_map
    @TinyJit
    def tiny_function(*args:Tensor):
      outs = gm(*[wrap(x) for x in args])
      # Filter out None values (can happen in backward pass)
      for x in outs:
        if x is not None: unwrap(x).realize()
      return outs
    # TODO: this should be able to pass in .tiny() Tensors, not need to convert them. it tries to access Storage if you pass in.
    def torch_function(*args:torch.Tensor):
      result = tiny_function(*[unwrap(x.tiny()) for x in args])
      # Convert results back to CPU to match input device for autograd
      if isinstance(result, (list, tuple)):
        return type(result)(r.cpu() if r is not None and hasattr(r, 'cpu') else r for r in result)
      return result.cpu() if result is not None and hasattr(result, 'cpu') else result
    return torch_function
  return aot_module_simplified(gm, sample_inputs, decompositions={}, fw_compiler=my_compiler)



if __name__ == "__main__":
  if getenv("TINY_BACKEND"):
    import tinygrad.nn.torch  # noqa: F401
    # When using torch.compile with tiny backend, keep everything on CPU
    # The backend will handle conversion to tinygrad internally
    device = torch.device("cpu")
  else:
    device = torch.device({"METAL":"mps","NV":"cuda"}.get(Device.DEFAULT, "cpu"))
  if DEBUG >= 1: print(f"using torch backend {device}")
  X_train, Y_train, X_test, Y_test = mnist()
  X_train = torch.tensor(X_train.float().numpy(), device=device)
  Y_train = torch.tensor(Y_train.cast(dtypes.int64).numpy(), device=device)
  X_test = torch.tensor(X_test.float().numpy(), device=device)
  Y_test = torch.tensor(Y_test.cast(dtypes.int64).numpy(), device=device)

  if getenv("TORCHVIZ"): torch.cuda.memory._record_memory_history()
  model = Model().to(device)
  optimizer = optim.Adam(model.parameters(), 1e-3)

  loss_fn = nn.CrossEntropyLoss()

  # Use "tiny" if we have "TINY_BACKEND" otherwise the default PyTorch's "inductor"
  compile_backend = "tiny" if getenv("TINY_BACKEND") else "inductor"

  # Compile only the forward pass computation
  @torch.compile(backend=compile_backend)
  def forward_and_loss(X, Y):
    out = model(X)
    loss = loss_fn(out, Y)
    return loss

  def step(samples):
    X,Y = X_train[samples], Y_train[samples]
    loss = forward_and_loss(X, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

  test_acc = float('nan')
  for i in (t:=trange(getenv("STEPS", 70))):
    samples = torch.randint(0, X_train.shape[0], (512,))  # putting this in JIT didn't work well
    loss = step(samples)
    if i%10 == 9: test_acc = ((model(X_test).argmax(axis=-1) == Y_test).sum() * 100 / X_test.shape[0]).item()
    t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")

  # verify eval acc
  if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
    if test_acc >= target and test_acc != 100.0: print(colored(f"{test_acc=} >= {target}", "green"))
    else: raise ValueError(colored(f"{test_acc=} < {target}", "red"))
  if getenv("TORCHVIZ"):
    torch.cuda.memory._dump_snapshot(fp:=temp("torchviz.pkl", append_user=True))
    print(f"saved torch memory snapshot to {fp}, view in https://pytorch.org/memory_viz")
