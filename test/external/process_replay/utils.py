import difflib, logging
from tinygrad.helpers import colored, getenv

def print_diff(s0, s1, unified=getenv("UNIFIED_DIFF",1)):
  if unified:
    lines = list(difflib.unified_diff(str(s0).splitlines(), str(s1).splitlines()))
    diff = "\n".join(colored(line, "red" if line.startswith("-") else "green" if line.startswith("+") else None) for line in lines)
  else:
    import ocdiff
    diff = ocdiff.console_diff(str(s0), str(s1))
  logging.info(diff)
