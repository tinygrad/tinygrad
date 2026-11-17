from __future__ import annotations
import sys, signal, argparse, bisect, codecs, itertools, typing, re, string, unicodedata, json, time, pathlib, threading
from typing import TYPE_CHECKING
from tinygrad import nn
from tinygrad.uop.ops import UOp, Ops
from tinygrad.helpers import partition, DEBUG, Timing, GlobalCounters, Context, fetch, profile_marker, getenv, round_up
from tinygrad.llm.model import Transformer
if TYPE_CHECKING:
  import jinja2

# (the conversations' texts and tokens SimpleTokenizer.encode keeps: one an agent of the server's (96 of them: their prompts not encoded again))
ENC_CACHE = getenv("ENC_CACHE", 256)
# (in a vocabulary with GLM's <|user|> and <|assistant|>, a role marker a reply generates opens the next turn: it ends the reply (the GGUF's
# eot / eom are <|user|> / <|observation|>; after a short answer the model sometimes writes <|assistant|> and a second turn after it))
ROLE_MARKERS = ("<|system|>", "<|user|>", "<|assistant|>", "<|observation|>")

class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="llama3",
               bos_id:int|None=None, eos_id:int=0, eot_id:int|None=None, eom_id:int|None=None, chat_template:str|None=None,
               control_ids:frozenset[int]=frozenset()):
    # (text, its tokens, (each special token's end, the tokens to it)) (see encode)
    self._enc_cache: list[tuple[str, list[int], list[tuple[int, int]]]] = []
    self._enc_lock = threading.Lock()
    preset = {"qwen35":"qwen2","qwen35moe":"qwen2","minimax-m2":"gpt-4o","deepseek-v3":"joyai-llm"}.get(preset, preset)
    if preset not in ("llama3","llama-v3","llama-bpe","qwen2","olmo","kimi-k2","tekken","glm4","gpt-4o","joyai-llm"):
      raise ValueError(f"Invalid tokenizer preset '{preset}'")
    # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]  # bytes that map to themselves
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i,b in enumerate(b for b in range(256) if b not in bs)}

    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L286
    # each limit is one past the category's max codepoint (Z→U+3000, N→U+1FBF9, L→U+323AF, M→U+E01EF, P→U+1E95F, S→U+1FBCA)
    # compact adjacent codepoints into ranges: listing them all makes re spend seconds on large prompts
    def ucat_range(pre:str|tuple[str, ...]) -> str:
      limits = {"Z": 0x3001, "N": 0x1fbfa, "L": 0x323b0, "M": 0xe01f0, "P": 0x1e960, "S": 0x1fbcb}
      limit = max(limits[p if p in limits else p[0]] for p in (pre if isinstance(pre, tuple) else (pre,)))
      cps = enumerate(cp for cp in range(limit) if unicodedata.category(chr(cp)).startswith(pre))
      runs = [list(g) for _, g in itertools.groupby(cps, lambda e: e[1]-e[0])]
      return "".join(re.escape(chr(g[0][1])) + (f"-{re.escape(chr(g[-1][1]))}" if len(g) > 1 else "") for g in runs)
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    contr, r_l, r_n = "(?i:'s|'t|'re|'ve|'m|'ll|'d)", f"[^\\r\\n{r_p_N}{r_p_L}]?", f"[{r_p_N}]" if preset == "tekken" else f"[{r_p_N}]{{1,3}}"
    r_p, r_w, r_t = f" ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*", f"{contr}|{r_l}[{r_p_L}]+", f"[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+"
    if preset in ("tekken", "gpt-4o", "kimi-k2"):
      r_up, r_lo = ucat_range(("Lu","Lt","Lm","Lo","M")), ucat_range(("Ll","Lm","Lo","M"))
      sfx = "" if preset == "tekken" else f"{contr}?"
      r_w = f"{r_l}[{r_up}]*[{r_lo}]+{sfx}|{r_l}[{r_up}]+[{r_lo}]*{sfx}"
      if preset != "kimi-k2": r_p = f" ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n/]*"
    self._pre_split = None
    if preset == "joyai-llm":
      r_ps = ucat_range(("P", "S"))
      r_p, r_w = f" ?[{r_ps}]+[\\r\\n]*", f"[{re.escape(string.punctuation)}][A-Za-z]+|[^\\r\\n{r_p_L}{r_ps}]?[{r_p_L}{ucat_range('M')}]+"
      # numbers and CJK runs are split off before the words
      self._pre_split = re.compile(f"({r_n}|[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+)")
    self._split_to_word = re.compile(f"({r_w}|{r_n}|{r_p}|{r_t})")
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")

    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self.preset, self.chat_template = preset, chat_template
    self.bos_id, self.eos_id, self.eot_id, self.eom_id = bos_id, eos_id, eot_id, eom_id
    self._ends = {eos_id, eot_id, eom_id} | \
      ({special_tokens[t] for t in ROLE_MARKERS if t in special_tokens} if all(t in special_tokens for t in ROLE_MARKERS[1:3]) else set())
    # (the CONTROL tokens (GGUF token type 3): text no reply shows)
    self.control_ids = control_ids

  @staticmethod
  def from_gguf_kv(kv:dict):
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] not in (2, 3, 4)) # UNKNOWN|CONTROL|USER_DEFINED
    special_tokens_dict = dict(special_tokens)
    control = frozenset(i for i, t in enumerate(kv["tokenizer.ggml.token_type"]) if t == 3)
    return SimpleTokenizer(dict(normal_tokens), special_tokens_dict, kv["tokenizer.ggml.pre"], control_ids=control,
      bos_id=kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None,
      eos_id=kv.get('tokenizer.ggml.eos_token_id', 0), eot_id=kv.get('tokenizer.ggml.eot_token_id', special_tokens_dict.get('<|im_end|>')),
      # (a GGUF without an eom: <|endoftext|> ends a reply too, as llama.cpp's end-of-generation set and the models' generation_config
      # (Qwen 3.x: eos <|im_end|> 248046 and <|endoftext|> 248044): else a reply the model ends with it decodes on into '<|im_start|>...')
      eom_id=kv.get('tokenizer.ggml.eom_token_id', special_tokens_dict.get('<|endoftext|>')), chat_template=kv.get('tokenizer.chat_template'))

  @staticmethod
  def from_hf(path:str|pathlib.Path, preset:str):
    path = pathlib.Path(path)
    cfg, tok = (json.loads((path / name).read_text()) for name in ("config.json", "tokenizer.json"))
    tok_cfg = json.loads((path / "tokenizer_config.json").read_text()) if (path / "tokenizer_config.json").exists() else {}
    added = {t["content"]: t["id"] for t in tok.get("added_tokens", [])}
    vocab = tok["model"]["vocab"]
    eos = cfg.get("eos_token_id", 0)
    eos = eos if isinstance(eos, list) else [eos]
    return SimpleTokenizer({t: i for t, i in vocab.items() if t not in added}, added, preset,
      bos_id=vocab.get(tok_cfg.get("bos_token")), eos_id=eos[0],
      eot_id=eos[1] if len(eos) > 1 else None, eom_id=eos[2] if len(eos) > 2 else None,
      chat_template=template.read_text() if (template:=path / "chat_template.jinja").exists() else tok_cfg.get("chat_template"))

  def _encode_word(self, word:bytes) -> list[int]:
    if (early_token:=self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [bytes([b]) for b in word]
    # greedily merge any parts that we can
    while True:
      i = min([(sys.maxsize, -1)] + [(self._normal_tokens.get(parts[j]+parts[j+1], sys.maxsize), j) for j in range(len(parts)-1)])[1]
      if i == -1: break
      parts[i:i+2] = [parts[i] + parts[i+1]]
    try: return [self._normal_tokens[p] for p in parts]
    except KeyError: raise RuntimeError("token not found")
  def _encode_sentence(self, chunk:str) -> list[int]:
    if self._pre_split is None: words = self._split_to_word.findall(chunk)
    else: words = [w for p in self._pre_split.split(chunk) for w in self._split_to_word.split(p) if w]
    return [tok for word in words for tok in self._encode_word(word.encode())]
  def encode(self, text:str) -> list[int]:
    """
    the tokens of text. the text up to its last special token is kept with its tokens (the last ENC_CACHE ones): a text that starts with one encodes
    only what follows (the text between special tokens is encoded on its own: the same tokens). a conversation's prompts grow. one that leaves a
    kept text part way (a template that renders older turns otherwise once a user message follows them: Qwen preserve_thinking=False, GLM
    clear_thinking, every Pokemon step) starts at the kept text's last special token before they differ (its tokens there: the ends kept); the
    kept one goes if they share more than half of it (the conversation's previous prompt)
    """
    with self._enc_lock:
      base, tokens, ends, src = "", [], [], None
      for t, ids, e in self._enc_cache[::-1]:
        if len(t) <= len(base) or not text.startswith(t[:(k:=min(4096, len(t)))]): continue
        if text.startswith(t): base, tokens, ends, src = t, ids, e, None
        else:
          # (the characters they share, then the last special token's end within them, past the best start yet)
          lo, hi = k, min(len(t), len(text))
          while lo < hi:
            if text[:(mid:=(lo + hi + 1) // 2)] == t[:mid]: lo = mid
            else: hi = mid - 1
          if (i:=bisect.bisect_right(e, (lo, sys.maxsize))) and e[i-1][0] > len(base):
            base, tokens, ends, src = t[:e[i-1][0]], ids[:e[i-1][1]], e[:i], (t if e[i-1][0] * 2 > len(t) else None)
    tokens, ends, pos, cut, cut_n = list(tokens), list(ends), len(base), None, 0
    for match in self._split_to_sentence.finditer(text, pos):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = cut = match.end(0)
      cut_n = len(tokens)
      ends.append((cut, cut_n))
    if cut is not None:
      with self._enc_lock:
        # (a conversation's own each: a server's clients (agents and their side requests) are dozens, a 250k token prompt encoded again takes
        # seconds. one is a few MB)
        keep = [e for e in self._enc_cache if not text[:cut].startswith(e[0]) and e[0] is not src]
        self._enc_cache = keep[len(keep)-(ENC_CACHE-1):] + [(text[:cut], tokens[:cut_n], ends)]
    return tokens + self._encode_sentence(text[pos:])

  def decode(self, ids:list[int]) -> str: return b''.join(self._tok2bytes[tid] for tid in ids).decode(errors='replace')
  def stream_decoder(self) -> typing.Callable[..., str]:
    dec = codecs.getincrementaldecoder('utf-8')('replace')
    def _decode(tid:int|None=None) -> str: return dec.decode(self._tok2bytes[tid]) if tid is not None else dec.decode(b'', final=True)
    return _decode
  def is_end(self, token_id:int) -> bool: return token_id in self._ends

models = {
  "llama3.2:1b": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf",
  "llama3.2:1b-q4": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
  "llama3.2:3b": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf",
  "llama3.2:3b-f16": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf",
  "llama3.1:8b": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
  "qwen3:0.6b": "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
  "qwen3:1.7b": "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
  "qwen3:8b": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
  "qwen3:30b-a3b": "https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf",
  "qwen3.5:0.8b": "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf",
  "qwen3.5:4b": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
  "qwen3.5:9b": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
  "qwen3.6:27b": "https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/resolve/main/Qwen3.6-27B-Q4_K_M.gguf",
  "qwen3.6:35b-a3b": "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
  # pinned to the last revision with the plain IQ4_XS quant (the replacement uses mixed UD quantization)
  "qwen3.8:27b": "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/b62a80264f8b0c1bb849ee1c9c487415ebeca194/Qwen3.8-27B-IQ4_XS.gguf",
  "olmoe": "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
  "moonlight": "https://huggingface.co/gabriellarson/Moonlight-16B-A3B-Instruct-GGUF/resolve/main/Moonlight-16B-A3B-Instruct-Q4_K_M.gguf",
  "glm-4.7-flash": "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf",
}

def chat_template(ct:str, tok:SimpleTokenizer) -> jinja2.Template:
  """a model's chat template (jinja2; an ImportError without it)"""
  import jinja2
  dict_attrs = frozenset(dir(dict))
  class ChatEnvironment(jinja2.Environment):
    # (message.role of a plain dict: its item at once, the value jinja2's getattr gives after the attribute lookup failed (a dict's own
    # attributes, items / get / ..., still first): the same text, without an AttributeError raised a message's every field (a Pokemon prompt
    # of ~1,800 messages: ~80K of them))
    def getattr(self, obj:typing.Any, attribute:str) -> typing.Any:
      if type(obj) is dict and attribute not in dict_attrs:
        try: return obj[attribute]
        except KeyError: return self.undefined(obj=obj, name=attribute)
      return super().getattr(obj, attribute)
  # chat templates are written for the environment of HF apply_chat_template: the newline after a block tag and the whitespace before it
  # aren't output (a model generates what it was trained on: re-rendered history then matches its tokens, its cache is reused)
  env = ChatEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols'])  # (for {% break %})
  env.filters['tojson'] = lambda obj, **kwargs: json.dumps(obj, **kwargs)  # jinja2's tojson escapes <>& for HTML safety
  env.globals['raise_exception'] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
  env.globals['strftime_now'] = lambda fmt: time.strftime(fmt)
  env.globals['bos_token'] = tok.decode([tok.bos_id]) if tok.bos_id is not None else ""
  env.globals['eos_token'] = tok.decode([tok.eos_id])
  return env.from_string(ct)

class FallbackTemplate:
  # minimal jinja2.Template-compatible chat template without jinja2, no tool calling support
  def __init__(self, tok:SimpleTokenizer): self.tok = tok
  def role(self, role:str) -> str:
    if self.tok.preset == 'olmo': return "<|" + role + "|>\n"  # OLMoE Instruct format
    if self.tok.preset == 'kimi-k2': return "<|im_" + role + "|>" + role + "<|im_middle|>"
    if self.tok.preset == 'qwen2': return "<|im_start|>" + role + "\n"
    if self.tok.preset == 'glm4': return "<|" + role + "|>"
    if self.tok.preset == 'tekken':
      if role == 'user': return "[INST]"
      if role == 'assistant': return ""
      raise ValueError(f"Unsupported role '{role}' for tokenizer preset '{self.tok.preset}'")
    return "<|start_header_id|>" + role + "<|end_header_id|>\n\n"
  def end_turn(self) -> str:
    if self.tok.preset == 'olmo': return "\n"
    if self.tok.preset == 'kimi-k2': return self.tok.decode([self.tok.eos_id])
    if self.tok.preset == 'qwen2': return self.tok.decode([self.tok.eos_id]) + "\n"
    if self.tok.preset == 'glm4': return ""
    if self.tok.preset == 'tekken': return "[/INST]"
    return self.tok.decode([self.tok.eos_id])
  def render(self, messages:list[dict], tools=None, add_generation_prompt:bool=True, preserve_thinking:bool=False, **kwargs) -> str:
    out = self.tok.decode([] if self.tok.bos_id is None else [self.tok.bos_id]) + ("<sop>" if self.tok.preset == 'glm4' else "")
    for msg in messages:
      out += self.role(msg["role"])
      content = msg.get("content")
      if isinstance(content, str): out += content
      elif isinstance(content, list):
        for c in content:
          if c["type"] == "text": out += c["text"]
          else: raise RuntimeError(f"unhandled type: {c['type']}")
      elif content is not None: raise RuntimeError(f"unknown content type: {type(content)}")
      out += self.end_turn()
    return out + self.role("assistant") if add_generation_prompt else out

from tinygrad.llm.serve import LLMServer

def server_sampling(kv:dict) -> dict[str, float]:
  """a request's temperature and top_p without its own: the model's recommended ones (general.sampling.* of its gguf; none: greedy, top_p 1)"""
  return {"temperature": float(kv.get("general.sampling.temp", 0.0)), "top_p": float(kv.get("general.sampling.top_p", 1.0))}

def _production_vision_projector_missing(architecture:str) -> bool:
  """Qwen and GLM Flash production have no text-only mode. C9 and unknown-model text runs have no mandatory tower."""
  if architecture not in ("qwen4exp", "glm5-next"): return False
  if all(d.strip().split(":", 1)[0].upper() in {"NULL", "CPU"} for d in getenv("DEV", "").split(",")):
    return False
  if not (mmproj := getenv("VISION_MMPROJ", "")): return True
  try:
    path = pathlib.Path(mmproj)
    return not path.is_file() or path.stat().st_size == 0
  except OSError: return True

STOP_DEADLINE_S = 60  # (a clean GLM Flash exit took 18 s on g3; the launchers wait 90 s for a stop)
def _on_sigterm(*_):
  import faulthandler
  faulthandler.dump_traceback_later(STOP_DEADLINE_S, exit=True)
  # (the compile workers got the same signal: a pool left to its exit finalizer waits for a dead worker's lock)
  from tinygrad.engine.worker import kill_worker_pool
  kill_worker_pool()
  sys.exit(0)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", "-m", default=list(models.keys())[0],
                      help=f"Model choice ({', '.join(models.keys())}) or local GGUF file/HF checkpoint directory")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  parser.add_argument("--serve", nargs='?', type=int, const=8000, metavar="PORT", help="Run OpenAI compatible API (optional port, default 8000)")
  parser.add_argument("--warmup", action="store_true", help="warmup the JIT")
  parser.add_argument("--benchmark", nargs='?', type=lambda s: tuple(int(x) for x in (s if "," in s else f"0,{s}").split(",", 1)), const=(0, 20),
                      metavar="[PREFILL,]DECODE", help="Benchmark tok/s (PREFILL prompt tokens then DECODE new tokens, default 0,20)")
  parser.add_argument("--no_chat_template", action="store_true", help="Don't use the model's chat template, always use the fallback template")
  parser.add_argument("--shard", type=int, default=1, help="Tensor parallel device count")
  args = parser.parse_args()
  # (MEM_TRACE=1, a tree with extra/: the AM driver's live VRAM allocations by call site after the warmup (the server's workspace), or at an OOM)
  if (mem_trace:=getenv("MEM_TRACE")):
    import extra.mem_trace as mt
    mt.install()
  if getenv("DEBUG_LOAD"):
    import os
    print(f"load: main at epoch {time.time():.2f} (the process made at epoch {os.stat('/proc/self').st_ctime:.2f})", flush=True)
  # (SIGTERM exits as an interrupt does: the devices are finalized (am: their queues stopped, the clocks at their lowest), not left pinned high.
  # a stop ends the server: if the exit itself hangs, every thread's stack goes to the log and the process ends STOP_DEADLINE_S after the signal)
  signal.signal(signal.SIGTERM, _on_sigterm)
  if args.benchmark: args.max_context = max(args.max_context, round_up(sum(args.benchmark)+1, 1024))

  # load the model
  st = time.perf_counter()
  with Context(DEBUG=max(DEBUG.value, 1 if args.serve else 0)):
    path = fetch(models.get(args.model, args.model))
    model, kv = (Transformer.from_hf if path.is_dir() else Transformer.from_gguf)(path, args.max_context, shard=args.shard)
  if _production_vision_projector_missing(kv.get("general.architecture", "")):
    parser.error("VISION_MMPROJ must name a nonempty projector file for Qwen and GLM Flash production; "
                 "unset is only for explicit DEV=NULL or DEV=CPU tests")
  model_name = kv.get('general.name') or kv.get('general.basename') or kv.get('_name_or_path') or args.model
  file_sizes = [y.nbytes()*args.shard for y in UOp.sink(*[x.uop for x in nn.state.get_parameters(model)]).toposort() if y.op is Ops.BUFFER]
  print(f"loaded model \"{model_name}\" at {sum(file_sizes)*1e-9/(time.perf_counter()-st):.2f} GB/s with {sum(file_sizes):,} bytes "
        f"and {sum(x.numel() for x in nn.state.get_parameters(model)):,} params, "
        f"max context {args.max_context} on {nn.state.get_parameters(model)[0].device}")

  # get tokenizer
  tok = SimpleTokenizer.from_hf(path, preset='glm4') if path.is_dir() else SimpleTokenizer.from_gguf_kv(kv)

  # use the model's chat template if jinja2 is available (enables model-specific formatting)
  template: jinja2.Template|FallbackTemplate = FallbackTemplate(tok)
  if not args.no_chat_template and (ct := tok.chat_template) is not None:
    try:
      template = chat_template(ct, tok)
    except ImportError: print("warning: jinja2 is not installed, the model's chat template is disabled")

  # VISION_MMPROJ: Qwen / GLM Flash's mmproj GGUF (their sequence engines). GLM-5.3 itself has no compatible vision tower,
  # and leaves this empty. Data-URL images are seen by supported models (see serve.extract_pictures). Load before caches are made.
  vision = None
  def load_vision(mmproj:str):
    from tinygrad.llm import qwen_vision
    from tinygrad.llm.qwen_vision import VisionTower, VISION_MAX_TOKENS
    from tinygrad.llm.gguf import gguf_load
    from tinygrad.helpers import to_tuple
    # (the tower of the mmproj's projector: qwen3vl_merger (Qwen 3.8 Flash) or glm5v (GLM 5.3 Flash: glm_vision, its picture sizing))
    tower:type[VisionTower] = VisionTower
    if gguf_load(mmproj)[0].get("clip.projector_type") == "glm5v":
      from tinygrad.llm.glm_vision import GlmVisionTower
      tower, qwen_vision.SIZING[0] = GlmVisionTower, "glm5v"
    # (its jits' intermediates in the model's scratch: the tower runs between the model's jits, in its thread)
    vision = tower.from_gguf(mmproj, devices=to_tuple(typing.cast(str|tuple[str, ...], model.token_embd.weight.device)),
                                   scratch=getattr(model, "_scratch", None))
    # (every bucket's jit compiled and its buffers made before serving: VISION_MAX_TOKENS)
    with Timing(f"vision: {mmproj} on {len(vision.devices)} devices, its jits up to {VISION_MAX_TOKENS} rows a picture: "):
      vision.warmup(VISION_MAX_TOKENS)
    return vision
  if (mmproj:=getenv("VISION_MMPROJ", "")):
    assert getattr(model, "seqs_supported", lambda: False)() and getattr(model, "image_pad_id", -1) >= 0, "VISION: qwen4exp's sequence engine"
    # (decoding the pictures: pillow in the server's environment, said at the start; 10.0.1 and later have libwebp's CVE-2023-4863 fixed)
    import PIL
    assert tuple(int(x) for x in re.findall(r"\d+", PIL.__version__)[:3]) >= (10, 0, 1), f"VISION: pillow {PIL.__version__}, 10.0.1 or later needed"
    setattr(model, "vision", True)
    vision = load_vision(mmproj)

  # warmup the JIT
  if args.warmup or args.serve:
    try:
      with Context(DEBUG=max(DEBUG.value, 1)): model.warmup()

      # (VISION: the model's captures grew the scratch the tower shares: its jits linked again by a blank picture of every bucket)
      if vision is not None:
        from tinygrad.llm.qwen_vision import VISION_MAX_TOKENS
        with Timing("vision: every bucket run again after the model's warm-up: "): vision.run_again(VISION_MAX_TOKENS)
    except MemoryError:
      if mem_trace: mt.report("oom")
      raise
    if mem_trace: mt.report("warm")
  if mmproj and vision is None: vision = load_vision(mmproj)

  # start server
  if args.serve:
    print(f"ready: serving on port {args.serve}", flush=True)
    sampling = server_sampling(kv)
    print(f"sampling: a request without its own gets temperature {sampling['temperature']} top_p {sampling['top_p']} "
          f"(the gguf's general.sampling.*; none: greedy)", flush=True)
    LLMServer(('', args.serve), model, model_name, tok, template, **sampling, vision=vision).serve_forever()

  # do benchmark
  if args.benchmark is not None:
    keep, (n_pp, n_tg) = 20, args.benchmark
    gen = model.generate(toks:=tok.encode(" ".join(map(str, range(n_pp))))[:n_pp] if n_pp else [tok.bos_id or 0])
    for i in range(n_tg+1):
      profile_marker(f"decode @ {i-1}" if i else "prefill")
      if i == keep+1 and n_tg > 2*keep: print("...")
      GlobalCounters.reset()
      with Timing(on_exit=lambda x: f", {(max(n_pp, 1) if i == 0 else 1)*1e9/x:8.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s,"
                  f" {GlobalCounters.global_mem//1000000}/{GlobalCounters.mem_used//1000000} MB  --  "+\
                  tok.decode(toks[n_pp if i <= keep or n_tg <= 2*keep else max(n_pp, 1)+n_tg-keep+1:]).replace("\n", "\\n"),
                  prefix=f"prefill {max(n_pp, 1):5d}: " if i == 0 else f"decode {i-1:6d}: ", enabled=i <= keep or i > n_tg-keep):
        next(gen)
    exit(0)

  # interactive chat
  messages: list[dict] = []
  while 1:
    try: messages.append({"role":"user", "content":input('>>> ')})
    except EOFError: break
    ids = tok.encode(template.render(messages=messages, add_generation_prompt=True))
    reply, dec = "", tok.stream_decoder()
    for next_id in model.generate(ids):
      if tok.is_end(next_id):
        sys.stdout.write(dec() + "\n\n")
        break
      reply += (piece := dec(next_id))
      sys.stdout.write(piece)
      sys.stdout.flush()
    messages.append({"role":"assistant", "content":reply})

if __name__ == "__main__": main()
