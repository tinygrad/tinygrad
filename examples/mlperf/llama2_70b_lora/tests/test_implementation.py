#!/usr/bin/env python3
"""Test script for Llama2 70B LoRA MLPerf implementation"""

import sys
from pathlib import Path
from typing import Callable, Final, List

# Add tinygrad root to path
sys.path.insert(0, str(Path(__file__).parents[3]))

TEST_SEPARATOR: Final[str] = "=" * 50
SUCCESS_MARKER: Final[str] = "SUCCESS"
FAILURE_MARKER: Final[str] = "FAILURE"
CHECK_MARK: Final[str] = "✓"
CROSS_MARK: Final[str] = "✗"

REQUIRED_LORA_CONFIG_KEYS: Final[List[str]] = ['r', 'alpha', 'dropout', 'target_modules']
REQUIRED_ROUGE_KEYS: Final[List[str]] = ['rouge-1', 'rouge-2', 'rouge-l']

TEST_LORA_INPUT_DIM: Final[int] = 512
TEST_LORA_OUTPUT_DIM: Final[int] = 256
TEST_LORA_RANK: Final[int] = 16
TEST_LORA_ALPHA: Final[float] = 32.0
TEST_BATCH_SIZE: Final[int] = 2
TEST_SEQUENCE_LENGTH: Final[int] = 10

TEST_PREDICTION: Final[str] = "the quick brown fox"
TEST_REFERENCE: Final[str] = "the quick brown fox jumps"
TEST_TEXT: Final[str] = "Hello world this is a test"


class TestResult:
  """Container for test execution results"""
  
  def __init__(self, *, name: str, passed: bool, message: str):
    self.name = name
    self.passed = passed
    self.message = message
  
  def display(self) -> None:
    """Display test result with appropriate formatting"""
    status_symbol = CHECK_MARK if self.passed else CROSS_MARK
    print(f"{status_symbol} {self.message}")


class ImportTester:
  """Handles module import testing"""
  
  @staticmethod
  def test_lora_import() -> TestResult:
    """Test LoRA module import"""
    try:
      from examples.mlperf.llama2_70b_lora.lora import LoRALinear, get_lora_config
      return TestResult(
        name="lora_import",
        passed=True,
        message="LoRA module imported successfully"
      )
    except ImportError as e:
      return TestResult(
        name="lora_import",
        passed=False,
        message=f"Failed to import LoRA module: {e}"
      )
  
  @staticmethod
  def test_rouge_import() -> TestResult:
    """Test ROUGE module import"""
    try:
      from examples.mlperf.llama2_70b_lora.train import compute_rouge_scores
      return TestResult(
        name="rouge_import",
        passed=True,
        message="ROUGE module imported successfully"
      )
    except ImportError as e:
      return TestResult(
        name="rouge_import",
        passed=False,
        message=f"Failed to import ROUGE module: {e}"
      )
  
  @staticmethod
  def test_dataset_import() -> TestResult:
    """Test dataset module import"""
    try:
      from examples.mlperf.llama2_70b_lora.dataset import GovReportDataset, SimpleTokenizer
      return TestResult(
        name="dataset_import",
        passed=True,
        message="Dataset module imported successfully"
      )
    except ImportError as e:
      return TestResult(
        name="dataset_import",
        passed=False,
        message=f"Failed to import Dataset module: {e}"
      )


class FunctionalityTester:
  """Handles functionality testing"""
  
  @staticmethod
  def test_lora_forward_pass() -> TestResult:
    """Test LoRA layer forward pass"""
    try:
      from examples.mlperf.llama2_70b_lora.lora import LoRALinear
      from tinygrad import Tensor
      
      lora_layer = LoRALinear(
        in_features=TEST_LORA_INPUT_DIM,
        out_features=TEST_LORA_OUTPUT_DIM,
        r=TEST_LORA_RANK,
        alpha=TEST_LORA_ALPHA
      )
      
      input_tensor = Tensor.randn(TEST_BATCH_SIZE, TEST_SEQUENCE_LENGTH, TEST_LORA_INPUT_DIM)
      output_tensor = lora_layer(input_tensor)
      
      expected_shape = (TEST_BATCH_SIZE, TEST_SEQUENCE_LENGTH, TEST_LORA_OUTPUT_DIM)
      
      if output_tensor.shape == expected_shape:
        return TestResult(
          name="lora_forward",
          passed=True,
          message="LoRA forward pass works correctly"
        )
      else:
        return TestResult(
          name="lora_forward",
          passed=False,
          message=f"LoRA output shape incorrect: {output_tensor.shape}"
        )
    except Exception as e:
      return TestResult(
        name="lora_forward",
        passed=False,
        message=f"LoRA test failed: {e}"
      )
  
  @staticmethod
  def test_rouge_computation() -> TestResult:
    """Test ROUGE score computation"""
    try:
      from examples.mlperf.llama2_70b_lora.train import compute_rouge_scores
      
      predictions = [TEST_PREDICTION]
      references = [TEST_REFERENCE]
      
      scores = compute_rouge_scores(predictions, references)
      
      if all(key in scores for key in REQUIRED_ROUGE_KEYS):
        rouge_1_f = scores['rouge-1']['f']
        return TestResult(
          name="rouge_computation",
          passed=True,
          message=f"ROUGE computation works (ROUGE-1 F1: {rouge_1_f:.3f})"
        )
      else:
        return TestResult(
          name="rouge_computation",
          passed=False,
          message="ROUGE scores missing expected keys"
        )
    except Exception as e:
      return TestResult(
        name="rouge_computation",
        passed=False,
        message=f"ROUGE test failed: {e}"
      )
  
  @staticmethod
  def test_tokenizer() -> TestResult:
    """Test tokenizer functionality"""
    try:
      from examples.mlperf.llama2_70b_lora.dataset import SimpleTokenizer
      
      tokenizer = SimpleTokenizer()
      tokens = tokenizer.encode(TEST_TEXT)
      
      if isinstance(tokens, list) and len(tokens) > 0:
        return TestResult(
          name="tokenizer",
          passed=True,
          message=f"Tokenizer works (tokens: {len(tokens)})"
        )
      else:
        return TestResult(
          name="tokenizer",
          passed=False,
          message="Tokenizer failed to produce tokens"
        )
    except Exception as e:
      return TestResult(
        name="tokenizer",
        passed=False,
        message=f"Dataset test failed: {e}"
      )


class ConfigurationTester:
  """Handles configuration testing"""
  
  @staticmethod
  def test_lora_config() -> TestResult:
    """Test LoRA configuration loading"""
    try:
      from examples.mlperf.llama2_70b_lora.lora import get_lora_config
      
      config = get_lora_config()
      
      if all(key in config for key in REQUIRED_LORA_CONFIG_KEYS):
        config_details = [
          f"  Rank: {config['r']}",
          f"  Alpha: {config['alpha']}",
          f"  Target modules: {config['target_modules']}"
        ]
        
        message = "LoRA configuration loaded successfully\n" + "\n".join(config_details)
        
        return TestResult(
          name="lora_config",
          passed=True,
          message=message
        )
      else:
        return TestResult(
          name="lora_config",
          passed=False,
          message="LoRA configuration missing required keys"
        )
    except Exception as e:
      return TestResult(
        name="lora_config",
        passed=False,
        message=f"Configuration test failed: {e}"
      )


class TestSuite:
  """Main test suite orchestrator"""
  
  def __init__(self):
    self.test_functions: List[Callable[[], TestResult]] = [
      ImportTester.test_lora_import,
      ImportTester.test_rouge_import,
      ImportTester.test_dataset_import,
      FunctionalityTester.test_lora_forward_pass,
      FunctionalityTester.test_rouge_computation,
      FunctionalityTester.test_tokenizer,
      ConfigurationTester.test_lora_config
    ]
  
  def run_all_tests(self) -> int:
    """Execute all tests and return exit code"""
    self._display_header()
    
    results = self._execute_tests()
    passed_count = sum(1 for result in results if result.passed)
    total_count = len(results)
    
    self._display_results(passed_count=passed_count, total_count=total_count)
    
    return 0 if passed_count == total_count else 1
  
  def _display_header(self) -> None:
    """Display test suite header"""
    print(TEST_SEPARATOR)
    print("MLPerf Llama2 70B LoRA Implementation Tests")
    print(TEST_SEPARATOR)
  
  def _execute_tests(self) -> List[TestResult]:
    """Execute all test functions"""
    results: List[TestResult] = []
    
    print("Testing imports...")
    for i, test_func in enumerate(self.test_functions[:3]):
      result = test_func()
      result.display()
      results.append(result)
    
    print("\nTesting LoRA functionality...")
    result = self.test_functions[3]()
    result.display()
    results.append(result)
    
    print("\nTesting ROUGE functionality...")
    result = self.test_functions[4]()
    result.display()
    results.append(result)
    
    print("\nTesting dataset functionality...")
    result = self.test_functions[5]()
    result.display()
    results.append(result)
    
    print("\nTesting configuration...")
    result = self.test_functions[6]()
    result.display()
    results.append(result)
    
    return results
  
  def _display_results(self, *, passed_count: int, total_count: int) -> None:
    """Display final test results"""
    print(f"\n{TEST_SEPARATOR}")
    print(f"Test Results: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
      print(f"{SUCCESS_MARKER} All tests passed! Implementation is ready.")
    else:
      print(f"{FAILURE_MARKER} Some tests failed. Please check the implementation.")


def main() -> int:
  """Main entry point for test execution"""
  test_suite = TestSuite()
  return test_suite.run_all_tests()


if __name__ == "__main__":
  sys.exit(main())