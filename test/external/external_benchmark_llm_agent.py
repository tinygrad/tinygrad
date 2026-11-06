"""Coding-agent workload data and a bounded-step driver for full-model, long-context measurements.

Use AgentBenchmark(model, tokenizer) in a resident process. prepare() resets prompt reuse; prefill_step()
advances one ordinary model chunk, and decode_step() consumes an EOS-aware response window. Compilation
is included unless the caller explicitly warms the corresponding captures first. Chunk timings exclude
RPC gaps: use prefill_all() for end-to-end warm prefill timing, once the individual steps are validated.
"""
import array, ast, hashlib, time
from pathlib import Path
from tinygrad import Device, UOp
from tinygrad.llm.qwen4exp import _host_ints

TASKS = {
  "sampling-review": """Review a proposed change in tinygrad/llm/model.py that removes the temperature-mode suffix from the JIT cache keys:

    # in Transformer.__call__:
    name = 'jits_dense' if dense else 'jits'
    # in Transformer._spec_call:
    name = '_spec_jit_dense' if dense else '_spec_jit'

The author says temperature is already an argument, so distinct greedy and stochastic captures are redundant.
Check this against the actual sampling and generation implementation in the repository snapshot. Explain whether this is safe,
provide the correct minimal patch if needed, and write runnable regression tests alternating greedy and nonzero temperatures,
including speculative decoding. Do not replace real functions with made-up APIs. Include the patch and tests, not just a plan.""",
  "loader-regression": """Investigate a proposed simplification of tinygrad/llm/gguf.py: replace _preadv_exact's retry loop with one
os.preadv call and ignore its return value. The author tested it on a local SSD and says partial reads cannot happen.
Use the actual implementation and its callers from the snapshot to assess the change. Provide a safe implementation and runnable
regression tests for short reads, EOF after partial progress, and correct offset advancement. Explain resource cleanup on failure.
Keep the existing public API, and provide concrete Python code and a minimal unified diff.""",
  "cache-debug": """A developer reports that after a long prompt, switching to another conversation with a shared prefix occasionally
produces a continuation of the previous conversation. This model has recurrent KDA state, attention KV caches, and speculative MTP.
Trace prefix reuse, checkpoint save/restore, and speculative state through tinygrad/llm/model.py. Identify the invariants a fix must
preserve. Propose a focused regression test that distinguishes valid prefix reuse from invalid recurrent-state reuse. Use actual
method names and explain exactly which cached state each assertion checks. Supply runnable test code, not pseudocode; do not invent
a bug if the current implementation already enforces the required invariant.""",
}


def coding_prompt(tokenizer, task:str, tokens:int, root:Path|None=None) -> list[int]:
  """Exact-sized prompt, filled with actual repository source rather than repeated text or random token IDs."""
  root = root or Path(__file__).resolve().parents[2]
  prefix = tokenizer.encode("<|im_start|>system\nYou are a coding agent working on tinygrad. Use the supplied repository sources. "
                            "Be precise about correctness and provide concrete changes and tests.<|im_end|>\n<|im_start|>user\n"
                            "Repository snapshot follows. Its final file may be truncated by the context budget.\n")
  suffix = tokenizer.encode("\nEND REPOSITORY SNAPSHOT\n\n" + TASKS[task] +
                            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
  budget = tokens - len(prefix) - len(suffix)
  if budget < 1: raise ValueError("context budget cannot fit the task")
  priority = ["tinygrad/llm/model.py", "tinygrad/llm/gguf.py", "test/runtime/test_llm_sampling.py",
              "test/runtime/test_llm_mtp_cache.py", "tinygrad/llm/serve.py", "tinygrad/engine/jit.py"]
  paths = [root / p for p in priority]
  paths += [p for folder in ("tinygrad", "test") for p in sorted((root / folder).rglob("*.py")) if p not in paths]
  context = []
  relevant = {"sampling-review": ("_sample", "__call__", "_spec_call", "generate"),
              "loader-regression": ("_preadv_exact",), "cache-debug": ("_reuse", "_checkpoint", "generate")}[task]
  source_path = root / ("tinygrad/llm/gguf.py" if task == "loader-regression" else "tinygrad/llm/model.py")
  source = source_path.read_text()
  tree = ast.parse(source)
  scope = tree if task == "loader-regression" else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Transformer")
  for name in relevant:
    node = next(n for n in scope.body if isinstance(n, ast.FunctionDef) and n.name == name)
    context.extend(tokenizer.encode(f"\n--- {source_path.relative_to(root)}:{node.lineno} ({name}) ---\n" +
                                    ast.get_source_segment(source, node)))
  for path in paths:
    context.extend(tokenizer.encode(f"\n--- {path.relative_to(root)} ---\n" + path.read_text()))
    if len(context) >= budget: break
  if len(context) < budget: raise ValueError("not enough distinct repository source for the requested context")
  return prefix + context[:budget] + suffix


class AgentBenchmark:
  def __init__(self, model, tokenizer): self.model, self.tokenizer = model, tokenizer

  def prepare(self, task:str, tokens:int, prompt:list[int]|None=None):
    m = self.model
    assert 0 < tokens <= m.max_context
    self.prompt = coding_prompt(self.tokenizer, task, tokens) if prompt is None else prompt[:]
    assert len(self.prompt) == tokens
    self.position, self.prefill_seconds, self.output, self.ended, self.gen = 0, 0., [], False, None
    self.end_reason = None
    m._cached_tokens, m._ckpt_tokens, m.spec_stats = [], [], [0, 0]
    self.devices = m.token_embd.weight.device
    assert isinstance(self.devices, tuple) and m.config.spec_drafts > 0
    for d in self.devices: Device[d].synchronize()
    if not hasattr(m, "_token_buf"): m._token_host, m._token_buf = _host_ints(self.devices, m.max_context+1)
    m._token_host[:] = array.array('i', self.prompt + [-1] * (m.max_context+1-len(self.prompt)))
    m._token_prev = self.prompt[:]
    m._init_states()
    m._spec_init()
    m._spec_row = 0
    m._spec_row0(0)
    self.sizes = m.prefill_chunks()
    self.start_var = UOp.variable("start_pos", 0, m.max_context-1)
    self.tok_vars = {n: UOp.variable("toks", 1, n) for n in self.sizes}
    return {"task": task, "prompt_tokens": len(self.prompt), "max_context": m.max_context,
            "prompt_sha256": hashlib.sha256(array.array('i', self.prompt).tobytes()).hexdigest()}

  def prefill_step(self):
    m = self.model
    n = min(self.sizes[-1], len(self.prompt)-self.position)
    if not n: raise ValueError("prefill is complete")
    sp = self.start_var.bind(self.position)
    nt = self.tok_vars[next(s for s in self.sizes if n <= s)].bind(n)
    begin = time.perf_counter()
    m(m._token_buf[:, sp:sp+nt], sp, None).realize()
    for d in self.devices: Device[d].synchronize()
    seconds = time.perf_counter()-begin
    self.position += n
    self.prefill_seconds += seconds
    return {"position": self.position, "tokens": n, "seconds": seconds, "tok_s": n/seconds,
            "complete": self.position == len(self.prompt)}

  def prefill_all(self):
    begin, count = time.perf_counter(), len(self.prompt)-self.position
    while self.position < len(self.prompt): self.prefill_step()
    seconds = time.perf_counter()-begin
    return {"tokens": count, "seconds": seconds, "tok_s": count/seconds}

  def decode_step(self, count:int=128):
    m = self.model
    assert self.position == len(self.prompt), "finish the actual prefill before decoding"
    if self.gen is None:
      if len(self.prompt) >= m.max_context: raise ValueError("no output headroom")
      m._checkpoint(self.prompt, save=True)
      first = m._wait_host(lambda: m._token_host[len(self.prompt)], "agent benchmark first token")
      self.output = [first]
      self.ended = self.tokenizer.is_end(first)
      if self.ended: self.end_reason = "eos"
      tokens = self.prompt + [first]
      m._cached_tokens, m._spec_row = self.prompt[:], 0
      m._token_prev = tokens
      self.gen = m._spec_steps(tokens, None)
    begin, actual = time.perf_counter(), 0
    for _ in range(count):
      if self.ended: break
      try: token = next(self.gen)
      except StopIteration:
        self.ended, self.end_reason = True, "context_limit"
        break
      self.output.append(token)
      if self.tokenizer.is_end(token):
        self.ended, self.end_reason = True, "eos"
        break
      actual += 1
    seconds = time.perf_counter()-begin
    return {"tokens": actual, "seconds": seconds, "tok_s": actual/seconds, "ended": self.ended, "end_reason": self.end_reason,
            "context_tokens": len(self.prompt)+len(self.output), "spec_stats": list(m.spec_stats),
            "text": self.tokenizer.decode(self.output)}
