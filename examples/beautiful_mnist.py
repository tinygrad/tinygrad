from tinygrad import Tensor, TinyJit, nn
from extra.datasets import fetch_mnist   # TODO: from tinygrad.extra.datasets should fix PYTHONPATH issues
from tqdm import trange

class Model:
  def __init__(self):
    self.c1 = nn.Conv2d(1, 8, (3,3))
    self.c2 = nn.Conv2d(8, 16, (3,3))
    self.l1 = nn.Linear(400, 10)
  def __call__(self, x:Tensor) -> Tensor:
    x = self.c1(x).relu().max_pool2d()
    x = self.c2(x).relu().max_pool2d()
    return self.l1(x.flatten(1)).log_softmax()   # TODO: there's a second one of these in sparse_categorical_crossentropy

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = fetch_mnist()

  model = Model()
  opt = nn.optim.Adam(nn.state.get_parameters(model))

  # TODO: there's a compiler error if you comment out TinyJit since randint isn't being realized and there's something weird with int
  @TinyJit
  def train_step(samples:Tensor) -> Tensor:
    opt.zero_grad()
    loss = model(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).backward()
    opt.step()
    return loss.realize()  # TODO: should the jit do this automatically? i think yes

  for i in (t:=trange(200)):
    samples = Tensor.randint(32, high=X_train.shape[0])  # TODO: put this in the JIT when rand is fixed
    loss = train_step(samples)
    t.set_description(f"loss: {loss.item():.2f}")

  test_acc = (model(X_test).argmax(axis=1) == Y_test).mean()
  print(f"Accuracy: {test_acc.item()*100:.2f}%")
