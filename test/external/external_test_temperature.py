import unittest
from tinygrad import Tensor
from tinygrad.apps.llm import sample_with_temperature, Transformer

class TestTemperature(unittest.TestCase):
  def test_greedy_sampling(self):
    logits = Tensor([1.0, 2.0, 5.0, 3.0, 1.5])
    results = [int(sample_with_temperature(logits, 0.0).item()) for _ in range(5)]
    self.assertTrue(all(r == results[0] for r in results))
    self.assertEqual(results[0], 2)

  def test_temperature_zero_threshold(self):
    logits = Tensor([1.0, 2.0, 5.0, 3.0])
    result = sample_with_temperature(logits, 1e-7)
    self.assertEqual(int(result.item()), 2)

  def test_sampling_variety(self):
    logits = Tensor([3.0, 3.1, 3.2, 3.0])
    results = [int(sample_with_temperature(logits, 1.0).item()) for _ in range(20)]
    unique_results = set(results)
    self.assertGreater(len(unique_results), 1)
    self.assertTrue(all(0 <= r < 4 for r in results))

  def test_temperature_distribution(self):
    logits = Tensor([1.0, 1.0, 4.0, 1.0])

    def get_best_ratio(temperature, samples=50):
      results = [int(sample_with_temperature(logits, temperature).item()) for _ in range(samples)]
      return sum(1 for r in results if r == 2) / len(results)

    low_temp_ratio = get_best_ratio(0.1)
    high_temp_ratio = get_best_ratio(2.0)
    self.assertGreater(low_temp_ratio, high_temp_ratio)

  def test_tensor_shape(self):
    logits = Tensor([1.0, 2.0, 3.0])
    result = sample_with_temperature(logits, 0.5)
    self.assertEqual(result.shape, (1,))

  def test_single_choice(self):
    logits = Tensor([5.0])
    result = sample_with_temperature(logits, 1.0)
    self.assertEqual(int(result.item()), 0)

  def test_extreme_temperatures(self):
    logits = Tensor([1.0, 2.0, 3.0])

    result_low = sample_with_temperature(logits, 1e-8)
    self.assertEqual(int(result_low.item()), 2)

    result_high = sample_with_temperature(logits, 100.0)
    self.assertIn(int(result_high.item()), [0, 1, 2])

  def test_model_integration(self):
    import inspect
    forward_sig = inspect.signature(Transformer.forward)
    self.assertIn('temperature', forward_sig.parameters)

    generate_sig = inspect.signature(Transformer.generate)
    self.assertIn('temperature', generate_sig.parameters)
    self.assertEqual(generate_sig.parameters['temperature'].default, 0.0)

class TestTemperatureStatistics(unittest.TestCase):
  def test_deterministic_behavior(self):
    logits = Tensor([1.0, 3.0, 2.0, 4.0])
    results = [int(sample_with_temperature(logits, 0.0).item()) for _ in range(10)]
    self.assertTrue(all(r == 3 for r in results))

  def test_random_behavior(self):
    logits = Tensor([1.0, 3.0, 2.0, 4.0])
    results = [int(sample_with_temperature(logits, 1.0).item()) for _ in range(50)]
    unique_count = len(set(results))
    self.assertGreater(unique_count, 1)

  def test_temperature_scaling(self):
    logits = Tensor([1.0, 3.0, 2.0, 4.0])
    temperatures = [0.1, 0.5, 1.0, 2.0]
    ratios = []

    for temp in temperatures:
      results = [int(sample_with_temperature(logits, temp).item()) for _ in range(30)]
      best_ratio = sum(1 for r in results if r == 3) / len(results)
      ratios.append(best_ratio)

    for i in range(len(ratios) - 1):
      self.assertGreaterEqual(ratios[i], ratios[i + 1])

if __name__ == '__main__':
  unittest.main()