try:
  import torch
except ImportError:
  # PyTorch unavailable for running tests - prevents ImportError when dependent tests deselected.
  torch = None
