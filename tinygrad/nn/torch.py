# type: ignore
import sys, pathlib
sys.path.append(pathlib.Path(__file__).parent.parent.as_posix())
try:
  import extra.torch_backend.backend  # noqa: F401 # pylint: disable=unused-import
  try: import extra.torch_backend.compile  # noqa: F401 # pylint: disable=unused-import
  except ImportError: pass  # torch.compile stack is optional
except ImportError as e: raise ImportError("torch frontend not in release\nTo fix, install tinygrad from a git checkout with pip install -e .") from e
