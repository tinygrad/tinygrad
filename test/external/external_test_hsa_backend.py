import unittest
import subprocess


class TestHSABackend(unittest.TestCase):
  def test_llama_with_hsa(self):
    result = subprocess.run(
      [
        "python3",
        "examples/llama.py",
        "--temperature=0",
        "--count=50",
        '--prompt="Hello."',
        "--timing",
        "--shard",
        "2",
        "--size",
        "70B",
        "--gen",
        "2",
      ],
      capture_output=True,
      text=True,
    )
    self.assertEqual(result.returncode, 0)

    # Optional: Check output for expected values
    # self.assertIn("expected output", result.stdout)


if __name__ == "__main__":
  unittest.main()
