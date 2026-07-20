from __future__ import annotations
import sys, argparse, codecs, typing, re, unicodedata, json, time
from typing import TYPE_CHECKING
from tinygrad.helpers import BEAM, DEBUG, JIT_BATCH_SIZE, Timing, GlobalCounters, Context, fetch, profile_marker, getenv
from tinygrad.llm.model import Transformer
if TYPE_CHECKING:
  import jinja2

class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="llama3",
               bos_id:int|None=None, eos_id:int=0, eot_id:int|None=None):
    preset = {"qwen35":"qwen2","qwen35moe":"qwen2"}.get(preset, preset)
    if preset not in ("llama3","llama-v3","llama-bpe","qwen2","olmo","kimi-k2","tekken","glm4"):
      raise ValueError(f"Invalid tokenizer preset '{preset}'")
    # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]  # bytes that map to themselves
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i,b in enumerate(b for b in range(256) if b not in bs)}

    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L286
    # 0x323b0 is one past the max codepoint in unicode categories L/N/Z (0x323af is max L)
    # Compact adjacent codepoints into ranges. Build L/N/Z together: scanning Unicode three times is measurable at server startup.
    runs: dict[str, list[tuple[int, int]]] = {pre:[] for pre in "LNZ"}
    for cp in range(0x323b0):
      if (pre:=unicodedata.category(chr(cp))[0]) not in runs: continue
      if runs[pre] and cp == runs[pre][-1][1]+1: runs[pre][-1] = (runs[pre][-1][0], cp)
      else: runs[pre].append((cp, cp))
    def ucat_range(pre:str) -> str:
      def esc(cp:int) -> str: return f"\\U{cp:08x}"
      return "".join(esc(st) if st == en else f"{esc(st)}-{esc(en)}" for st,en in runs[pre])
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    self._split_to_word = re.compile("(?i:'s|'t|'re|'ve|'m|'ll|'d)|" + \
      f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+")
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")

    byte_translation = str.maketrans(self._byte_decoder)
    self._normal_tokens = {tok.translate(byte_translation).encode("latin1"): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self._encode_cache: tuple[str, tuple[int, ...], list[tuple[int, int]]]|None = None
    self.preset = preset
    self.bos_id, self.eos_id, self.eot_id = bos_id, eos_id, eot_id

  @staticmethod
  def from_gguf_kv(kv:dict):
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    normal_tokens: dict[str, int] = {}
    special_tokens: dict[str, int] = {}
    for idx,(tok,token_type) in enumerate(zip(kv["tokenizer.ggml.tokens"], kv["tokenizer.ggml.token_type"])):
      (normal_tokens if token_type == 1 else special_tokens)[tok] = idx
    return SimpleTokenizer(normal_tokens, special_tokens, kv["tokenizer.ggml.pre"],
      bos_id=kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None,
      eos_id=kv.get('tokenizer.ggml.eos_token_id', 0), eot_id=kv.get('tokenizer.ggml.eot_token_id'))

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
    return [tok for word in self._split_to_word.findall(chunk) for tok in self._encode_word(word.encode())]
  def encode(self, text:str) -> list[int]:
    tokens: list[int] = []
    pos = 0
    checkpoints: list[tuple[int, int]] = []
    if self._encode_cache is not None:
      old_text, old_tokens, old_checkpoints = self._encode_cache
      if text == old_text: return list(old_tokens)
      common, limit = 0, min(len(text), len(old_text))
      while common+4096 <= limit and text[common:common+4096] == old_text[common:common+4096]: common += 4096
      common += next((i for i,(a,b) in enumerate(zip(text[common:limit], old_text[common:limit])) if a != b), limit-common)
      if (checkpoint := next((x for x in reversed(old_checkpoints) if x[0] <= common), None)) is not None:
        pos, token_pos = checkpoint
        tokens, checkpoints = list(old_tokens[:token_pos]), [x for x in old_checkpoints if x[0] <= pos]
    for match in self._split_to_sentence.finditer(text, pos):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = match.end(0)
      checkpoints.append((pos, len(tokens)))
    tokens += self._encode_sentence(text[pos:])
    self._encode_cache = text, tuple(tokens), checkpoints
    return tokens

  def decode(self, ids:list[int]) -> str: return b''.join(self._tok2bytes[tid] for tid in ids).decode(errors='replace')
  def stream_decoder(self) -> typing.Callable[..., str]:
    dec = codecs.getincrementaldecoder('utf-8')('replace')
    def _decode(tid:int|None=None) -> str: return dec.decode(self._tok2bytes[tid]) if tid is not None else dec.decode(b'', final=True)
    return _decode
  def is_end(self, token_id:int) -> bool: return token_id in (self.eos_id, self.eot_id)

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
  "qwen3.5:27b": "https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/main/Qwen3.5-27B-Q4_K_M.gguf",
  "qwen3.5:35b-a3b": "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-Q4_K_M.gguf",
  "olmoe": "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
  "moonlight": "https://huggingface.co/gabriellarson/Moonlight-16B-A3B-Instruct-GGUF/resolve/main/Moonlight-16B-A3B-Instruct-Q4_K_M.gguf",
  "glm-4.7-flash": "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf",
}

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
  def render(self, messages:list[dict], tools=None, add_generation_prompt:bool=True, enable_thinking:bool=False,
             preserve_thinking:bool=False) -> str:
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

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", "-m", default=list(models.keys())[0], help=f"Model choice ({', '.join(models.keys())}) or path to a local GGUF file")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  parser.add_argument("--serve", nargs='?', type=int, const=8000, metavar="PORT", help="Run OpenAI compatible API (optional port, default 8000)")
  parser.add_argument("--warmup", action="store_true", help="warmup the JIT")
  parser.add_argument("--beam", type=int, help="Kernel optimization beam width")
  parser.add_argument("--benchmark", nargs='?', type=int, const=20, metavar="COUNT", help="Benchmark tok/s (optional count, default 20)")
  args = parser.parse_args()

  # load the model
  model_path = fetch(models.get(args.model, args.model))
  model, kv = Transformer.from_gguf(model_path, args.max_context)
  model_name = kv.get('general.name') or kv.get('general.basename') or args.model
  print(f"using model \"{model_name}\" with {model_path.stat().st_size:,} bytes and {model.parameter_count:,} params, "
        f"max context {model.max_context} on {model.token_embd.weight.device}")

  # get tokenizer
  tok = SimpleTokenizer.from_gguf_kv(kv)

  # use the model's chat template if jinja2 is available (enables model-specific formatting)
  template: jinja2.Template|FallbackTemplate = FallbackTemplate(tok)
  if (ct := kv.get('tokenizer.chat_template')) is not None:
    try:
      import jinja2
      env = jinja2.Environment()
      env.filters['tojson'] = lambda obj, **kwargs: json.dumps(obj, **kwargs)  # jinja2's tojson escapes <>& for HTML safety
      env.globals['raise_exception'] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
      env.globals['strftime_now'] = lambda fmt: time.strftime(fmt)
      env.globals['bos_token'] = tok.decode([tok.bos_id]) if tok.bos_id is not None else ""
      env.globals['eos_token'] = tok.decode([tok.eos_id])
      template = env.from_string(ct)
    except ImportError: print("warning: jinja2 is not installed, the model's chat template is disabled")

  # warmup the JIT
  if args.warmup or args.serve:
    amd_server = bool(args.serve) and str(model.token_embd.weight.device).startswith("AMD")
    beam = args.beam if args.beam is not None else 2 if amd_server else BEAM.value
    print(f"warming serving JITs with BEAM={beam}")
    batch_size = 448 if amd_server else JIT_BATCH_SIZE.value
    with Context(DEBUG=DEBUG.value, BEAM=beam, JIT_BATCH_SIZE=batch_size):
      model.warmup()

  # start server
  if args.serve: LLMServer(('', args.serve), model, model_name, tok, template).serve_forever()

  # do benchmark
  if args.benchmark is not None:
    gen = model.generate(toks:=[tok.bos_id or 0])
    for i in range(args.benchmark):
      profile_marker(f"decode @ {i}")
      GlobalCounters.reset()
      if (log:=getenv("BENCHMARK_LOG", "")): from extra.bench_log import WallTimeEvent, BenchEvent
      with Timing(on_exit=lambda x: f", {1e9/x:6.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s,"
                  f" {GlobalCounters.global_mem//1000000}/{GlobalCounters.mem_used//1000000} MB  --  "+\
                  tok.decode(toks).replace("\n", "\\n")):
        if log:
          with WallTimeEvent(BenchEvent.STEP): next(gen)
        else: next(gen)
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
