import unittest
from collections import OrderedDict, namedtuple
from types import SimpleNamespace
from tinygrad import Tensor
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict


class TestStateDict(unittest.TestCase):
  def test_container_subclasses(self):
    class TensorDict(dict): pass
    class TensorList(list): pass
    class TensorTuple(tuple): pass
    weight = Tensor([1., 2.])
    for container, key in ((TensorDict(weight=weight), "weight"), (TensorList([weight]), "0"), (TensorTuple([weight]), "0")):
      with self.subTest(container=type(container).__name__):
        container.description = "model weights"
        model = SimpleNamespace(layers=container)
        state = get_state_dict(model)
        self.assertEqual(list(state), [f"layers.{key}"])
        self.assertIs(state[f"layers.{key}"], weight)
        params = get_parameters(model)
        self.assertEqual(len(params), 1)
        self.assertIs(params[0], weight)

  def test_namedtuple_and_ordered_dict(self):
    first, second = Tensor([1.]), Tensor([2.])
    pair = namedtuple("Pair", ["first", "second"])(first, second)
    state = get_state_dict(OrderedDict(pair=pair))
    self.assertEqual(list(state), ["pair.first", "pair.second"])
    self.assertIs(state["pair.first"], first)
    self.assertIs(state["pair.second"], second)

  def test_load_container_subclass(self):
    class TensorDict(dict): pass
    weight = Tensor([1., 2.])
    model = TensorDict(weight=weight)
    loaded = load_state_dict(model, {"weight": Tensor([3., 4.])}, verbose=False)
    self.assertEqual(len(loaded), 1)
    self.assertIs(loaded[0], weight)
    self.assertEqual(weight.tolist(), [3., 4.])

  def test_container_tensor_attributes(self):
    class TensorDict(dict): pass
    class TensorList(list): pass
    class TensorTuple(tuple): pass
    for container_type in (TensorDict, TensorList, TensorTuple):
      with self.subTest(container=container_type.__name__):
        model = container_type()
        model.weight = Tensor([1., 2.])
        state = get_state_dict(model)
        self.assertEqual(list(state), ["weight"])
        self.assertIs(state["weight"], model.weight)
        params = get_parameters(model)
        self.assertEqual(len(params), 1)
        self.assertIs(params[0], model.weight)
        loaded = load_state_dict(model, {"weight": Tensor([3., 4.])}, verbose=False)
        self.assertEqual(len(loaded), 1)
        self.assertIs(loaded[0], model.weight)
        self.assertEqual(model.weight.tolist(), [3., 4.])

  def test_container_contents_and_attributes(self):
    class TensorDict(dict): pass
    class TensorList(list): pass
    class TensorTuple(tuple): pass
    item, weight = Tensor([1.]), Tensor([2.])
    for model, key in ((TensorDict(item=item), "item"), (TensorList([item]), "0"), (TensorTuple([item]), "0")):
      with self.subTest(container=type(model).__name__):
        model.weight = weight
        state = get_state_dict(model, prefix="model.")
        self.assertEqual(list(state), [f"model.{key}", "model.weight"])
        self.assertIs(state[f"model.{key}"], item)
        self.assertIs(state["model.weight"], weight)

  def test_container_attribute_precedence(self):
    class TensorDict(dict): pass
    model = TensorDict(weight=Tensor([1.]))
    model.weight = Tensor([2.])
    self.assertIs(get_state_dict(model)["weight"], model.weight)


if __name__ == '__main__':
  unittest.main()
