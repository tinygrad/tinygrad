# beautiful mnist in the new "one-shot" style
# one realize in the whole graph
# depends on:
#  - "big graph" UOp scheduling
#  - symbolic removal

from examples.beautiful_mnist import Model
from tinygrad import Tensor, nn, getenv
from tinygrad.nn.datasets import mnist

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = mnist()
  print("*** got data")

  model = Model()
  print("*** got model")

  opt = nn.optim.Adam(nn.state.get_parameters(model))
  print("*** got optimizer")

  samples = Tensor.randint(getenv("STEPS", 10), getenv("BS", 512), high=X_train.shape[0])
  X_samp, Y_samp = X_train[samples], Y_train[samples]
  print("*** got samples")

  with Tensor.train():
    # TODO: this shouldn't be a for loop. something like:
    """
    i = UOp.range(samples.shape[0])  # TODO: fix range function on UOp
    model(X_samp[i]).sparse_categorical_crossentropy(Y_samp[i]).backward()
    opt.schedule_steps(i)
    """
    for i in range(samples.shape[0]):
      opt.zero_grad()
      model(X_samp[i]).sparse_categorical_crossentropy(Y_samp[i]).backward()
      opt.schedule_step()
  print("*** scheduled training")

  # evaluate the model
  with Tensor.test():
    test_acc = ((model(X_test).argmax(axis=1) == Y_test).mean()*100)
  print("*** scheduled eval")

  # only actually do anything at the end
  print(f"test_accuracy: {test_acc.item():5.2f}%")
