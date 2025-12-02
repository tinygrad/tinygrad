import itertools
from examples.beautiful_mnist import Model
from tinygrad import nn, Tensor, dtypes, Device
from tinygrad.helpers import getenv, trange, partition

# TODO: refactor this into optim/onnx
def functional_adam(g:Tensor, m:Tensor, v:Tensor, b1_t:Tensor, b2_t:Tensor, lr=0.001, b1=0.9, b2=0.999, eps=1e-6) -> Tensor:
  b1_t *= b1
  b2_t *= b2
  m.assign(b1 * m + (1.0 - b1) * g)
  v.assign(b2 * v + (1.0 - b2) * (g * g))
  m_hat = m / (1.0 - b1_t)
  v_hat = v / (1.0 - b2_t)
  return lr * (m_hat / (v_hat.sqrt() + eps))

if __name__ == "__main__":
  BS = getenv("BS", 512)
  ACC_STEPS = getenv("ACC_STEPS", 4)

  X_train, Y_train, X_test, Y_test = nn.datasets.mnist()
  model = Model()

  params = nn.state.get_parameters(model)

  # set requires grad on the ones we need gradients of
  for x in params:
    if x.requires_grad is None: x.requires_grad_()

  # split params (with grads) and buffers (without)
  params, buffers = partition(nn.state.get_parameters(model), lambda x: x.requires_grad)
  print(f"params: {len(params)} buffers: {len(buffers)}")

  # optim params
  pos_params = list(itertools.accumulate(params, lambda x,y: x+y.numel(), initial=0))
  adam_m = Tensor.zeros(pos_params[-1], device="CPU").contiguous()
  adam_v = Tensor.zeros(pos_params[-1], device="CPU").contiguous()
  adam_b1_t = Tensor.ones((1,), dtype=dtypes.float32, device="CPU", requires_grad=False)
  adam_b2_t = Tensor.ones((1,), dtype=dtypes.float32, device="CPU", requires_grad=False)
  adam_params = [adam_m, adam_v, adam_b1_t, adam_b2_t]

  @Tensor.train()
  def microbatch(loss: Tensor, grads: Tensor):
    samples = Tensor.randint(BS // ACC_STEPS, high=X_train.shape[0])
    for t in params: t.grad = None
    # divide by ACC_STEPS at the loss
    uloss = (model(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]) / ACC_STEPS).backward()
    # concat the grads and assign them
    loss.assign(loss + uloss)
    grads.assign(grads + Tensor.cat(*[t.grad.contiguous().flatten() for t in params], dim=0))
    Tensor.realize(loss, grads)

  def get_test_acc() -> Tensor: return (model(X_test).argmax(axis=1) == Y_test).mean()*100

  test_acc = float('nan')
  for i in (t:=trange(getenv("STEPS", 70))):
    # microbatch sets the gradients
    loss = Tensor.zeros(tuple()).contiguous()
    grads = Tensor.zeros(pos_params[-1]).contiguous()
    for _ in range(ACC_STEPS): microbatch(loss, grads)

    # run optimizer (on CPU, where adam params live)
    delta = functional_adam(grads.to("CPU"), adam_m, adam_v, adam_b1_t, adam_b2_t)

    # update the params, copying back the delta one at a time to avoid OOM
    for j,tt in enumerate(params):
      tt.assign(tt.detach() - delta[pos_params[j]:pos_params[j+1]].reshape(tt.shape).to(Device.DEFAULT))

    # realize everything
    Tensor.realize(*params, *buffers, *adam_params)

    # eval
    if i%10 == 9: test_acc = get_test_acc().item()
    t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")
