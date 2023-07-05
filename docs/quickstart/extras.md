---
description: Some more advanced tips and tricks once you've got the quickstart running.
---

# Extras

## JIT

It is possible to speed up the computation of certain neural networks by using the JIT. Currently, this does not support models with varying input sizes and non-tinygrad operations.

To use the JIT we just need to add a function decorator to the forward pass of our neural network and ensure that the input and output are realized tensors. Or in this case, we will create a wrapper function and decorate the wrapper function to speed up the evaluation of our neural network.

```python
from tinygrad.jit import TinyJit

@TinyJit
def jit(x):
  return net(x).realize()

st = time.perf_counter()
avg_acc = 0
for step in range(1000):
  # random sample a batch
  samp = np.random.randint(0, X_test.shape[0], size=(64))
  batch = Tensor(X_test[samp], requires_grad=False)
  # get the corresponding labels
  labels = Y_test[samp]

  # forward pass with jit
  out = jit(batch)

  # calculate accuracy
  pred = np.argmax(out.numpy(), axis=-1)
  avg_acc += (pred == labels).mean()
print(f"Test Accuracy: {avg_acc / 1000}")
print(f"Time: {time.perf_counter() - st}")
```

You will find that the evaluation time is much faster than before and that your accelerator utilization is much higher.

## Saving and Loading Models

The standard weight format for tinygrad is [safetensors](https://github.com/huggingface/safetensors). This means that you can load the weights of any model also using safetensors into tinygrad. There are functions in [state.py](https://github.com/geohot/tinygrad/blob/master/tinygrad/state.py) to save and load models to and from this format.

```python
from tinygrad.state import safe_save, safe_load, get_state_dict, load_state_dict

# first we need the state dict of our model
state_dict = get_state_dict(net)

# then we can just save it to a file
safe_save(state_dict, "model.safetensors")

# and load it back in
state_dict = safe_load("model.safetensors")
load_state_dict(net, state_dict)
```

Many of the models in the [models/](https://github.com/geohot/tinygrad/blob/master/models) folder have a `load_from_pretrained` method that will download and load the weights for you. These usually are PyTorch weights meaning that you would need PyTorch installed to load them.

## Environment Variables

There exist a bunch of environment variables that control the runtime behavior of tinygrad. Some of the commons ones are `DEBUG` and the different backend enablement variables.

You can find a full list and their descriptions [here](../environment\_variables/).

## Visualizing the Computation Graph

It is possible to visualize the computation graph of a neural network using [graphviz](https://graphviz.org/).

This is easily done by running a single pass (forward or backward!) of the neural network with the environment variable `GRAPH` set to `1`. The graph will be saved to `/tmp/net.svg` by default.

To install graphviz:

{% tabs %}
{% tab title="macOS" %}
```bash
brew install graphviz
```

You may also need to install pydot

```bash
python3 -m pip install pydot
```
{% endtab %}
{% endtabs %}
