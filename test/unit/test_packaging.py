import pathlib, tomllib, unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]

class TestPackaging(unittest.TestCase):
  def test_every_package_dir_is_listed(self):
    # pyproject's package list is explicit: a subpackage it does not name is left out of the wheel.
    # tinygrad.llm.kernels was, and `python -m tinygrad.llm` failed to import from the 0.14.0 wheel.
    listed = set(tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["packages"])
    dirs = {py.parent for py in (ROOT / "tinygrad").rglob("*.py") if "__pycache__" not in py.parts}
    for d in sorted(dirs):
      self.assertIn(".".join(d.relative_to(ROOT).parts), listed, f"{d.relative_to(ROOT)} has .py files but is not in pyproject packages")

if __name__ == "__main__":
  unittest.main()
