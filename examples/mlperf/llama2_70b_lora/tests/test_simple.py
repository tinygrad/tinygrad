#!/usr/bin/env python3
"""Simple test script for Llama2 70B LoRA MLPerf implementation"""

import sys
from pathlib import Path
from typing import Callable, Final, List


SUCCESS_MARKER: Final[str] = "SUCCESS"
FAILURE_MARKER: Final[str] = "FAILURE"
CHECK_MARK: Final[str] = "✓"
CROSS_MARK: Final[str] = "✗"
TEST_SEPARATOR: Final[str] = "=" * 40

EXPECTED_LORA_CONFIG_KEYS: Final[List[str]] = ['r', 'alpha', 'dropout', 'target_modules']
TEST_PREDICTION: Final[str] = "the quick brown fox"
TEST_REFERENCE: Final[str] = "the quick brown fox jumps"
TEST_TEXT: Final[str] = "Hello world"


class TestResult:
  """Container for individual test execution results"""
  
  def __init__(self, *, name: str, passed: bool, message: str):
    self.name = name
    self.passed = passed
    self.message = message
  
  def display(self) -> None:
    """Display formatted test result"""
    status_symbol = CHECK_MARK if self.passed else CROSS_MARK
    print(f"{status_symbol} {self.message}")


class PathManager:
  """Manages Python path configuration for imports"""
  
  @staticmethod
  def configure_paths() -> None:
    """Configure Python paths for module imports"""
    current_dir = Path(__file__).parent  # tests/
    lora_dir = current_dir.parent        # llama2_70b_lora/
    tinygrad_root = current_dir.parent.parent.parent.parent  # tinygrad/
    
    sys.path.insert(0, str(lora_dir))     # Add llama2_70b_lora for direct imports
    sys.path.insert(0, str(tinygrad_root))  # Add tinygrad root


class LoRATestSuite:
  """Test suite for LoRA implementation"""
  
  @staticmethod
  def test_import() -> TestResult:
    """Test LoRA module import functionality"""
    try:
      import lora
      return TestResult(
        name="lora_import",
        passed=True,
        message="LoRA module imported successfully"
      )
    except ImportError as e:
      return TestResult(
        name="lora_import",
        passed=False,
        message=f"LoRA import failed: {e}"
      )
  
  @staticmethod
  def test_configuration() -> TestResult:
    """Test LoRA configuration retrieval"""
    try:
      import lora
      config = lora.get_lora_config()
      
      missing_keys = [key for key in EXPECTED_LORA_CONFIG_KEYS if key not in config]
      if missing_keys:
        return TestResult(
          name="lora_config",
          passed=False,
          message=f"LoRA config missing keys: {missing_keys}"
        )
      
      return TestResult(
        name="lora_config",
        passed=True,
        message=f"LoRA config valid: r={config['r']}, alpha={config['alpha']}"
      )
    except Exception as e:
      return TestResult(
        name="lora_config",
        passed=False,
        message=f"LoRA config test failed: {e}"
      )


class RougeTestSuite:
  """Test suite for ROUGE implementation"""
  
  @staticmethod
  def test_import() -> TestResult:
    """Test ROUGE module import functionality"""
    try:
      import train
      return TestResult(
        name="rouge_import",
        passed=True,
        message="ROUGE module imported successfully"
      )
    except ImportError as e:
      return TestResult(
        name="rouge_import",
        passed=False,
        message=f"ROUGE import failed: {e}"
      )
  
  @staticmethod
  def test_computation() -> TestResult:
    """Test ROUGE score computation"""
    try:
      import train
      
      predictions = [TEST_PREDICTION]
      references = [TEST_REFERENCE]
      
      scores = train.compute_rouge_scores(predictions=predictions, references=references)
      
      if 'rouge-1' not in scores or 'f' not in scores['rouge-1']:
        return TestResult(
          name="rouge_computation",
          passed=False,
          message="ROUGE scores missing expected structure"
        )
      
      rouge_1_f = scores['rouge-1']['f']
      return TestResult(
        name="rouge_computation",
        passed=True,
        message=f"ROUGE computation successful (ROUGE-1 F1: {rouge_1_f:.3f})"
      )
    except Exception as e:
      return TestResult(
        name="rouge_computation",
        passed=False,
        message=f"ROUGE computation failed: {e}"
      )


class DatasetTestSuite:
  """Test suite for dataset implementation"""
  
  @staticmethod
  def test_import() -> TestResult:
    """Test dataset module import functionality"""
    try:
      import dataset
      return TestResult(
        name="dataset_import",
        passed=True,
        message="Dataset module imported successfully"
      )
    except ImportError as e:
      return TestResult(
        name="dataset_import",
        passed=False,
        message=f"Dataset import failed: {e}"
      )
  
  @staticmethod
  def test_tokenizer() -> TestResult:
    """Test simple tokenizer functionality"""
    try:
      import dataset
      
      tokenizer = dataset.SimpleTokenizer()
      tokens = tokenizer.encode(text=TEST_TEXT)
      
      if not isinstance(tokens, list) or len(tokens) == 0:
        return TestResult(
          name="tokenizer_test",
          passed=False,
          message="Tokenizer returned invalid result"
        )
      
      return TestResult(
        name="tokenizer_test",
        passed=True,
        message=f"Tokenizer functional (tokens: {len(tokens)})"
      )
    except Exception as e:
      return TestResult(
        name="tokenizer_test",
        passed=False,
        message=f"Tokenizer test failed: {e}"
      )


class SimpleTestRunner:
  """Main test runner for simple implementation tests"""
  
  def __init__(self):
    self.test_suites: List[Callable[[], List[TestResult]]] = [
      self._run_lora_tests,
      self._run_rouge_tests,
      self._run_dataset_tests
    ]
  
  def _run_lora_tests(self) -> List[TestResult]:
    """Execute LoRA test suite"""
    print("Testing LoRA implementation...")
    return [
      LoRATestSuite.test_import(),
      LoRATestSuite.test_configuration()
    ]
  
  def _run_rouge_tests(self) -> List[TestResult]:
    """Execute ROUGE test suite"""
    print("\nTesting ROUGE implementation...")
    return [
      RougeTestSuite.test_import(),
      RougeTestSuite.test_computation()
    ]
  
  def _run_dataset_tests(self) -> List[TestResult]:
    """Execute dataset test suite"""
    print("\nTesting dataset implementation...")
    return [
      DatasetTestSuite.test_import(),
      DatasetTestSuite.test_tokenizer()
    ]
  
  def run_all_tests(self) -> int:
    """Execute all test suites and return exit code"""
    self._display_header()
    
    all_results: List[TestResult] = []
    for test_suite in self.test_suites:
      results = test_suite()
      all_results.extend(results)
      
      for result in results:
        result.display()
    
    return self._calculate_exit_code(results=all_results)
  
  def _display_header(self) -> None:
    """Display test execution header"""
    print(TEST_SEPARATOR)
    print("Simple Implementation Tests")
    print(TEST_SEPARATOR)
  
  def _calculate_exit_code(self, *, results: List[TestResult]) -> int:
    """Calculate appropriate exit code based on test results"""
    passed_count = sum(1 for result in results if result.passed)
    total_count = len(results)
    
    print(f"\nResults: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
      print(f"{SUCCESS_MARKER}: All basic tests passed!")
      return 0
    else:
      print(f"{FAILURE_MARKER}: Some tests failed.")
      return 1


def main() -> int:
  """Main entry point for simple test execution"""
  PathManager.configure_paths()
  
  test_runner = SimpleTestRunner()
  return test_runner.run_all_tests()


if __name__ == "__main__":
  sys.exit(main())