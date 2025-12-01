from tinygrad import dtypes, getenv, Device
from tinygrad.helpers import trange, colored, DEBUG, temp
from tinygrad.nn.datasets import mnist
import torch
from torch import nn, optim

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

if __name__ == "__main__":
  if getenv("TINY_BACKEND"):
    import tinygrad.nn.torch  # noqa: F401
    device = torch.device("tiny")
  else:
    device = torch.device({"METAL":"mps","NV":"cuda"}.get(Device.DEFAULT, "cpu"))
  if DEBUG >= 1: print(f"using torch backend {device}")
  X_train, Y_train, X_test, Y_test = mnist()

  # TORCH_COMPILE=1 enables torch.compile with tiny backend (requires CPU tensors for dynamo tracing)
  if getenv("TORCH_COMPILE") and getenv("TINY_BACKEND"):
    X_train = torch.tensor(X_train.float().numpy())
    Y_train = torch.tensor(Y_train.cast(dtypes.int64).numpy())
    X_test = torch.tensor(X_test.float().numpy())
    Y_test = torch.tensor(Y_test.cast(dtypes.int64).numpy())
    model = Model()
    compiled_model = torch.compile(model, backend="tiny")
  else:
    X_train = torch.tensor(X_train.float().numpy(), device=device)
    Y_train = torch.tensor(Y_train.cast(dtypes.int64).numpy(), device=device)
    X_test = torch.tensor(X_test.float().numpy(), device=device)
    Y_test = torch.tensor(Y_test.cast(dtypes.int64).numpy(), device=device)
    model = Model().to(device)
    compiled_model = model

  if getenv("TORCHVIZ"): torch.cuda.memory._record_memory_history()
  optimizer = optim.Adam(model.parameters(), 1e-3)
  loss_fn = nn.CrossEntropyLoss()

  def step(samples):
    X, Y = X_train[samples], Y_train[samples]
    out = compiled_model(X)
    Y = Y.to(out.device) if out.device.type != "cpu" else Y
    loss = loss_fn(out, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

  def eval_acc():
    if getenv("TORCH_COMPILE") and getenv("TINY_BACKEND"):
      # batch eval to match JIT cached shapes
      correct, bs = 0, 512
      with torch.no_grad():
        for i in range(0, X_test.shape[0], bs):
          batch_x, batch_y = X_test[i:i+bs], Y_test[i:i+bs]
          if batch_x.shape[0] < bs:
            batch_x = torch.cat([batch_x, torch.zeros(bs - batch_x.shape[0], *batch_x.shape[1:])])
            out = compiled_model(batch_x)[:batch_y.shape[0]]
          else:
            out = compiled_model(batch_x)
          correct += (out.argmax(axis=-1) == batch_y).sum().item()
      return correct * 100 / X_test.shape[0]
    return ((model(X_test).argmax(axis=-1) == Y_test).sum() * 100 / X_test.shape[0]).item()

  test_acc = float('nan')
  for i in (t:=trange(getenv("STEPS", 70))):
    samples = torch.randint(0, X_train.shape[0], (512,))
    loss = step(samples)
    if i%10 == 9: test_acc = eval_acc()
    t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")

  if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
    test_acc = eval_acc()
    if test_acc >= target and test_acc != 100.0: print(colored(f"{test_acc=} >= {target}", "green"))
    else: raise ValueError(colored(f"{test_acc=} < {target}", "red"))
  if getenv("TORCHVIZ"):
    torch.cuda.memory._dump_snapshot(fp:=temp("torchviz.pkl", append_user=True))
    print(f"saved torch memory snapshot to {fp}, view in https://pytorch.org/memory_viz")
