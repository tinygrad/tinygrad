from dataclasses import dataclass
import difflib, logging, subprocess, traceback
from typing import Dict, Optional
from tinygrad.helpers import ContextVar, colored, getenv

def print_diff(s0, s1, unified=getenv("UNIFIED_DIFF",1)):
  if not logging.getLogger().hasHandlers(): logging.basicConfig(level=logging.INFO, format="%(message)s")
  if unified:
    lines = list(difflib.unified_diff(str(s0).splitlines(), str(s1).splitlines()))
    diff = "\n".join(colored(line, "red" if line.startswith("-") else "green" if line.startswith("+") else None) for line in lines)
  else:
    import ocdiff
    diff = ocdiff.console_diff(str(s0), str(s1))
  logging.info(diff)

@dataclass(frozen=True)
class ProcessReplayContext:
  ctx_vars: Dict[str, int]
  loc: str = ""
  head_sha: Optional[str] = None
  run_id: Optional[int] = None
def get_process_replay_ctx() -> ProcessReplayContext:
  ctx_vars = {k:v.value for k,v in ContextVar._cache.items()}
  try: head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
  except Exception: head_sha = ""
  stack = filter(lambda x: not any(n in x.filename for n in ["engine/schedule.py", "engine/realize.py", "codegen/kernel.py", "unittest"]),
                 traceback.extract_stack()[:-1])
  loc = "\n".join(traceback.format_list(stack))
  return ProcessReplayContext(ctx_vars, loc, head_sha, getenv("GITHUB_RUN_ID") or None)
