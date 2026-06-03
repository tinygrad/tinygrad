from __future__ import annotations
import sys, argparse, codecs, typing, re, unicodedata, json, uuid, time, pathlib, cv2, math
from tinygrad import nn, Tensor, TinyJit, dtypes
from tinygrad.uop.ops import UOp, Ops
from tinygrad.helpers import partition, DEBUG, Timing, GlobalCounters, stderr_log, colored, Context, fetch, profile_marker
from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler
from tinygrad.llm.model import Transformer
from gguf import gguf_load
from tinygrad.nn.state import load_state_dict

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
    def ucat_range(pre: str): return "".join(re.escape(chr(cp)) for cp in range(0x323b0) if unicodedata.category(chr(cp)).startswith(pre))
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    self._split_to_word = re.compile("(?i:'s|'t|'re|'ve|'m|'ll|'d)|" + \
      f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+")
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")

    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self.preset = preset
    self.bos_id, self.eos_id, self.eot_id = bos_id, eos_id, eot_id

  @staticmethod
  def from_gguf_kv(kv:dict):
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] == 1)
    return SimpleTokenizer(dict(normal_tokens), dict(special_tokens), kv["tokenizer.ggml.pre"],
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
    for match in self._split_to_sentence.finditer(text):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = match.end(0)
    return tokens + self._encode_sentence(text[pos:])

  def decode(self, ids:list[int]) -> str: return b''.join(self._tok2bytes[tid] for tid in ids).decode(errors='replace')
  def stream_decoder(self) -> typing.Callable[..., str]:
    dec = codecs.getincrementaldecoder('utf-8')('replace')
    def _decode(tid:int|None=None) -> str: return dec.decode(self._tok2bytes[tid]) if tid is not None else dec.decode(b'', final=True)
    return _decode
  def role(self, role:str):
    if self.preset == 'olmo': return self.encode("<|" + role + "|>\n")  # OLMoE Instruct format
    if self.preset == 'kimi-k2': return self.encode("<|im_" + role + "|>" + role + "<|im_middle|>")
    if self.preset == 'qwen2': return self.encode("<|im_start|>" + role + "\n")
    if self.preset == 'glm4': return self.encode("<|" + role + "|>")
    if self.preset == 'tekken':
      if role == 'user': return self.encode("[INST]")
      if role == 'assistant': return []
      raise ValueError(f"Unsupported role '{role}' for tokenizer preset '{self.preset}'")
    return self.encode("<|start_header_id|>" + role + "<|end_header_id|>\n\n")
  def end_turn(self):
    if self.preset == 'olmo': return self.encode("\n")
    if self.preset == 'kimi-k2': return [self.eos_id]
    if self.preset == 'qwen2': return [self.eos_id] + self.encode("\n")
    if self.preset == 'glm4': return []
    if self.preset == 'tekken': return self.encode("[/INST]")
    return [self.eos_id]
  def prefix(self) -> list[int]:
    return ([] if self.bos_id is None else [self.bos_id]) + (self.encode("<sop>") if self.preset == 'glm4' else [])
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
  "qwen3:vl2b":"https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3VL-2B-Instruct-F16.gguf",
  "qwen3:vl4b":"https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen3VL-4B-Instruct-F16.gguf"
}

# *** simple OpenAI API compatible server with web interface on http://localhost:8000/ ***

class Handler(HTTPRequestHandler):
  server: LLMServer
  def log_request(self, code='-', size='-'): pass
  def do_GET(self):
    if self.path == "/v1/models": self.send_data(json.dumps({"object":"list","data":[{"id":self.server.model_name,"object":"model"}]}).encode())
    else: self.send_data((pathlib.Path(__file__).parent / "chat.html").read_bytes(), content_type="text/html")
  def run_model(self, ids:list[int], model_name:str, include_usage=False, max_tokens:int|None=None, temperature:float=0.0):
    model, tok = self.server.model, self.server.tok
    cache_start_pos = model.get_start_pos(ids)
    stderr_log(f"{self.path}  {colored('--', 'BLACK')}  "
               f"in:{colored(f'{cache_start_pos:5d}', 'green')} +{len(ids)-cache_start_pos:5d}  {colored('--', 'BLACK')}  ")
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    yield {"choices": [{"index":0, "delta":{"role":"assistant","content":""}, "finish_reason":None}], **tmpl}
    out: list[int] = []
    finish_reason = "stop"
    st = time.perf_counter()
    dec = tok.stream_decoder()
    for next_id in model.generate(ids, temperature=temperature):
      if len(out) == 0: stderr_log(f"prefill:{(len(ids)-cache_start_pos)/((pt:=time.perf_counter())-st):4.0f} tok/s  {colored('--', 'BLACK')}  ")
      if tok.is_end(next_id): break
      out.append(next_id)
      yield {"choices": [{"index":0, "delta":{"content":dec(next_id)}, "finish_reason":None}], **tmpl}
      if max_tokens is not None and len(out) >= max_tokens:
        finish_reason = "length"
        break
    if (tail := dec()): yield {"choices": [{"index":0, "delta":{"content":tail}, "finish_reason":None}], **tmpl}
    yield {"choices": [{"index":0, "delta":{},"finish_reason":finish_reason}], **tmpl}
    if include_usage:
      yield {"choices": [], "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}, **tmpl}
    et = time.perf_counter()
    stderr_log(f"gen:{len(out)/(et-pt) if len(out) > 1 else 0:4.0f} tok/s  {colored('--', 'BLACK')}  "
               f"out:{len(out):5d}  {colored('--', 'BLACK')}  total:{et-st:6.2f}s\n")

  def do_POST(self):
    tok = self.server.tok
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8"))
    if DEBUG >= 1: print(json.dumps(body, indent=2))
    if self.path == "/v1/chat/completions":
      # extract tokens, last assistant message is treated as prefill
      ids: list[int] = tok.prefix()
      for i, msg in enumerate(body["messages"]):
        ids += tok.role(msg["role"])
        content = msg["content"]
        if isinstance(content, str): ids += tok.encode(content)
        elif isinstance(content, list):
          for c in content:
            if c["type"] == "text": ids += tok.encode(c["text"])
            else: raise RuntimeError(f"unhandled type: {c['type']}")
        else: raise RuntimeError(f"unknown content type: {type(content)}")
        if msg["role"] == "assistant" and i == len(body["messages"]) - 1: break
        ids += tok.end_turn()
      else: ids += tok.role("assistant")

      # reply
      max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
      chunks = self.run_model(ids, body["model"], not body.get("stream") or body.get("stream_options",{}).get("include_usage", False),
                              max_tokens=max_tokens, temperature=float(body.get("temperature", 0.0)))
      if body.get("stream"): self.stream_json(chunks)
      else:
        out, finish_reason = [], "stop"
        for c in chunks:
          if c["choices"] and c["choices"][0].get("delta", {}).get("content"): out.append(c["choices"][0]["delta"]["content"])
          if c["choices"] and c["choices"][0].get("finish_reason"): finish_reason = c["choices"][0]["finish_reason"]
        self.send_data(json.dumps({**c, "object":"chat.completion",
          "choices":[{"index":0, "message":{"role":"assistant","content":"".join(out)}, "finish_reason":finish_reason}]}).encode())
    else:
      raise RuntimeError(f"unhandled path {self.path}")

class LLMServer(TCPServerWithReuse):
  def __init__(self, server_address:tuple, model:Transformer, model_name:str, tok:SimpleTokenizer):
    self.model, self.model_name, self.tok = model, model_name, tok
    super().__init__(server_address, Handler)

  
def prefill_img(vis, lang, image, start_pos, res=(640, 640)):
  if image.shape[:2] != res:
    target_h, target_w = res[:2]
    s = min(target_w / image.shape[1], target_h / image.shape[0])
    r = cv2.resize(image, (int(image.shape[1] * s), int(image.shape[0] * s)))
    image = cv2.copyMakeBorder(r, (target_h - r.shape[0]) // 2, target_h - r.shape[0] - (target_h - r.shape[0]) // 2, (target_w - r.shape[1]) // 2, target_w - r.shape[1] - (target_w - r.shape[1]) // 2, cv2.BORDER_CONSTANT, value=0)
  prefill(vis=vis, lang=lang, image=Tensor(image), start_pos=start_pos)

@TinyJit
def prefill(vis, lang, image, start_pos):
  image = image.permute(2, 0, 1)
  height, width = image.shape[-2:]
  image = image.unsqueeze(0).float()
  image = image.interpolate(size=(height, width))
  resized_height, resized_width = image.shape[-2:]
  patches = (image - 127.5) / 127.5
  batch_size, channel = 1, 3
  # https://github.com/huggingface/transformers/blob/4ae05b0fba41860adaaeb708774fc1f48c92c049/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L195
  grid_h, grid_w = resized_height // vis.patch_size, resized_width // vis.patch_size
  patches = patches.reshape(
      batch_size,
      channel,
      grid_h // vis.merge_size,
      vis.merge_size,
      vis.patch_size,
      grid_w // vis.merge_size,
      vis.merge_size,
      vis.patch_size,
  )
  patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
  pixel_values = (
      patches.unsqueeze(6)
      .expand(-1, -1, -1, -1, -1, -1, vis.temporal_patch_size, -1, -1)
      .reshape(
          batch_size,
          grid_h * grid_w,
          channel * vis.temporal_patch_size * vis.patch_size * vis.patch_size, # 1536
      )
  )[0]
  pixel_values = pixel_values.cast(dtypes.bfloat16)

  # f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n<|im_end|>\n" fill size of img with image token
  # <|im_end|>\n<|im_start|>assistant\n
  num_image_tokens = int((grid_h*grid_w) / 4)
  input_ids = Tensor.cat(Tensor([151644, 872, 198, 151652]), Tensor.zeros(num_image_tokens), Tensor([151653, 198, 151645, 198])).unsqueeze(0).cast(dtypes.int)

  image_embeds, hidden_states, deepstack_feature_lists = vis(pixel_values, [grid_h, grid_w])
  hidden_states = lang.token_embd(input_ids).cast(dtypes.float)
  # 4 to -4 because of <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n<|im_end|>\n tokens before and after image_pad
  hidden_states[:, 4:-4, :] = image_embeds.unsqueeze(0)
  
  # https://github.com/huggingface/transformers/blob/08692e3c31654e4825b4c078a3c70b86efa70a46/src/transformers/models/qwen3_vl/modular_qwen3_vl.py#L626
  # https://github.com/huggingface/transformers/blob/08692e3c31654e4825b4c078a3c70b86efa70a46/src/transformers/models/qwen3_vl/modular_qwen3_vl.py#L543
  for i in range(len(lang.blk)):
    hidden_states = lang.blk[i](hidden_states, start_pos=start_pos)
    # https://github.com/huggingface/transformers/blob/08692e3c31654e4825b4c078a3c70b86efa70a46/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L692
    if i in vis.v.deepstack_idx:
      # 4 to -4 because of <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n<|im_end|>\n tokens before and after image_pad
      hidden_states[:, 4:-4, :] += deepstack_feature_lists[vis.v.deepstack_idx.index(i)]
  hidden_states.realize()

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    ret = Tensor.cat(-x2, x1, dim=-1)
    return ret

class Qwen3VLVis():
  def __init__(self, size="2B"):
    kv, state_dict = gguf_load(fetch(f"https://huggingface.co/Qwen/Qwen3-VL-{size}-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-{size}-Instruct-F16.gguf"))
    self.merge_size = kv["clip.vision.spatial_merge_size"]
    self.patch_size = kv["clip.vision.patch_size"]
    self.temporal_patch_size = 2
    self.v = Qwen3VisBlocks(kv=kv, weights=state_dict)
    self.mm = [nn.Linear(*state_dict["mm.0.weight"].shape[::-1], bias=True), None, nn.Linear(*state_dict["mm.2.weight"].shape[::-1], bias=True)]
    state_dict["v.patch_embd.weight1"] = state_dict["v.patch_embd.weight.1"]
    load_state_dict(self, state_dict)
    self.inv_freq = 1.0 / (10000.0 ** (Tensor.arange(0, 32, 2, dtype=dtypes.float) / 32))

  # https://github.com/huggingface/transformers/blob/15bb519bd4277f4ab5309154aedf3c231e8b4ca8/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L679
  def __call__(self, pixel_values, image_grid_size):        
    grid_hs = image_grid_size[0]
    grid_ws = image_grid_size[1]

    h_idxs = Tensor.linspace(0, self.v.num_grid_per_side - 1, grid_hs)
    w_idxs = Tensor.linspace(0, self.v.num_grid_per_side - 1, grid_ws)

    h_idxs_floor = h_idxs.cast(dtypes.int32)
    w_idxs_floor = w_idxs.cast(dtypes.int32)
    h_idxs_ceil = (h_idxs_floor.int() + 1).clip(self.v.num_grid_per_side - 1)
    w_idxs_ceil = (w_idxs_floor.int() + 1).clip(self.v.num_grid_per_side - 1)
    dh = h_idxs - h_idxs_floor
    dw = w_idxs - w_idxs_floor

    base_h = h_idxs_floor * self.v.num_grid_per_side
    base_h_ceil = h_idxs_ceil * self.v.num_grid_per_side

    idx_tensor = Tensor.stack(
        (base_h[None].T + w_idxs_floor[None]).flatten(),
        (base_h[None].T + w_idxs_ceil[None]).flatten(),
        (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
        (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
    ).cast(dtypes.int32)

    weight_tensor = Tensor.stack(
        ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
        ((1 - dh)[None].T * dw[None]).flatten(),
        (dh[None].T * (1 - dw)[None]).flatten(),
        (dh[None].T * dw[None]).flatten(),
    ).cast(dtypes.bfloat16)

    pos_embeds = self.v.position_embd(idx_tensor)
    pos_embeds *= weight_tensor[:, :, None]
    pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

    merge_size = 2
    pos_embeds = (pos_embeds.view(1, grid_hs // merge_size, merge_size, grid_ws // merge_size, merge_size, -1).permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
    
    hpos_ids = Tensor.arange(grid_hs).unsqueeze(1).expand(-1, grid_ws)
    hpos_ids = hpos_ids.reshape(grid_hs // merge_size, merge_size, grid_ws // merge_size, merge_size).transpose(1, 2).flatten()

    wpos_ids = Tensor.arange(grid_ws).unsqueeze(0).expand(grid_hs, -1)
    wpos_ids = wpos_ids.reshape(grid_hs // merge_size, merge_size, grid_ws // merge_size, merge_size).transpose(1, 2).flatten()

    pos_ids = Tensor.stack(hpos_ids, wpos_ids, dim=-1).repeat(1, 1)

    rotary_pos_emb = (pos_ids.unsqueeze(-1) * self.inv_freq).flatten(1)

    hidden_states = pixel_values.view(-1, 3, 2, 16, 16)
    hidden_states = hidden_states.flatten(1, 2)
    w = Tensor.stack(self.v.patch_embd.weight, self.v.patch_embd.weight1, dim=2)
    out_C, in_C, kD, kH, kW = w.shape
    w2d = w.reshape(out_C, in_C * kD, kH, kW)

    hidden_states = hidden_states.conv2d(
        weight=w2d,
        bias=self.v.patch_embd.bias,
        stride=(16, 16),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1
    )

    hidden_states = hidden_states.view(-1, 1024)
    hidden_states = hidden_states + pos_embeds

    sqlen, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(sqlen, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(sqlen, -1)
    emb = Tensor.cat(rotary_pos_emb, rotary_pos_emb, dim=-1)
    cos, sin = emb.cos(), emb.sin()
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    
    deepstack_feature_lists = []
    for i in range(len(self.v.blk)):
      hidden_states = self.v.blk[i](hidden_states, cos, sin)
      if i in self.v.deepstack_idx: deepstack_feature_lists.append(self.v.deepstack[i](hidden_states))

    image_embeds = self.v.post_ln(hidden_states)
    image_embeds = image_embeds.view(-1, 4096)
    image_embeds = self.mm[0](image_embeds)
    image_embeds = Tensor.gelu(image_embeds)
    image_embeds = self.mm[2](image_embeds)
    return image_embeds, hidden_states, deepstack_feature_lists

class Qwen3PatchEmbed():
  def __init__(self, kv=None):
    self.weight = Tensor.zeros(kv["clip.vision.embedding_length"], 3, 16, 16)
    self.weight1 = Tensor.zeros(kv["clip.vision.embedding_length"], 3, 16, 16)
    self.bias = Tensor.zeros(kv["clip.vision.embedding_length"])
    
class Qwen3VisBlocks():
  def __init__(self, kv=None, weights=None):
    self.blk = []
    for _ in range(kv["clip.vision.block_count"]): self.blk.append(Qwen3VisBlock(kv, weights=weights))
    self.patch_embd = Qwen3PatchEmbed(kv=kv)
    self.num_grid_per_side = 48 # todo unhardcode
    self.deepstack_layers = kv["clip.vision.is_deepstack_layers"]
    self.deepstack_idx = [i for i, val in enumerate(self.deepstack_layers) if val]
    self.deepstack = []
    for i in range(len(self.deepstack_layers)):
      if i in self.deepstack_idx:
        self.deepstack.append(DeepstackLayer(i, weights))
      else:
        self.deepstack.append(None)
    self.position_embd = nn.Embedding(*weights["v.position_embd.weight"].shape)
    self.post_ln = nn.LayerNorm(weights["v.post_ln.weight"].shape[0], eps=1e-6, elementwise_affine=True)

class DeepstackLayer:
  def __init__(self, index, weights):
    self.fc1 = nn.Linear(*weights[f"v.deepstack.{index}.fc1.weight"].shape[::-1])
    self.fc2 = nn.Linear(*weights[f"v.deepstack.{index}.fc2.weight"].shape[::-1])
    self.norm = nn.LayerNorm(weights[f"v.deepstack.{index}.norm.weight"].shape[0], eps=1e-6, elementwise_affine=True)
    self.hidden_size = weights[f"v.deepstack.{index}.norm.weight"].shape[0]

  #https://github.com/huggingface/transformers/blob/027d1a97025295a1346c2eb5c361259e69eedfe7/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L112
  def __call__(self, hidden_states):
      deepstack_feature = (hidden_states.view(-1, self.hidden_size)).view(-1, self.hidden_size)
      return self.fc2(Tensor.gelu(self.fc1(deepstack_feature)))

class Qwen3VisBlock():
  def __init__(self, kv=None, weights=None):
    self.ffn_up = nn.Linear(kv["clip.vision.embedding_length"], kv["clip.vision.feed_forward_length"])
    self.ffn_down = nn.Linear(kv["clip.vision.feed_forward_length"], kv["clip.vision.embedding_length"])
    self.ln1 = nn.LayerNorm(kv["clip.vision.embedding_length"], eps=1e-6, elementwise_affine=True)
    self.ln2 = nn.LayerNorm(kv["clip.vision.embedding_length"], eps=1e-6, elementwise_affine=True)
    self.attn_out = nn.Linear(kv["clip.vision.embedding_length"], kv["clip.vision.embedding_length"])
    self.attn_qkv = nn.Linear(*weights["v.blk.0.attn_qkv.weight"].shape[::-1])
  
  def __call__(self, hidden_states, cos, sin):
    hidden_states_input = self.ln1(hidden_states)
    seq_length = hidden_states_input.shape[0]
    qkv = self.attn_qkv(hidden_states_input)
    qkv = qkv.reshape(seq_length, 3, 16, -1).permute(1, 0, 2, 3)
    query, key, value = qkv.chunk(3, dim=0)
    query = query.squeeze(0)
    key   = key.squeeze(0)
    value = value.squeeze(0)
    query = (query * cos) + (rotate_half(query) * sin)
    key = (key * cos) + (rotate_half(key) * sin)

    query = query.transpose(0, 1).unsqueeze(0)
    value = value.transpose(0, 1).unsqueeze(0)

    attn_weight = query @ key.transpose(0, 1).unsqueeze(0).transpose(-2, -1) * 0.125
    attn_weight = Tensor.softmax(attn_weight)
    attn_output = attn_weight @ value
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(seq_length, -1)
    attn_output = self.attn_out(attn_output)
    hidden_states += attn_output
    norm = self.ln2(hidden_states)
    norm = self.ffn_up(norm).gelu()
    norm = self.ffn_down(norm)
    return hidden_states + norm

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", "-m", default=list(models.keys())[0], help=f"Model choice ({', '.join(models.keys())}) or path to a local GGUF file")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  parser.add_argument("--serve", nargs='?', type=int, const=8000, metavar="PORT", help="Run OpenAI compatible API (optional port, default 8000)")
  parser.add_argument("--warmup", action="store_true", help="warmup the JIT")
  parser.add_argument("--benchmark", nargs='?', type=int, const=20, metavar="COUNT", help="Benchmark tok/s (optional count, default 20)")
  args = parser.parse_args()

  # load the model
  model, kv = Transformer.from_gguf(fetch(models.get(args.model, args.model)), args.max_context)
  model_name = kv.get('general.name') or kv.get('general.basename') or args.model
  file_sizes = [y.nbytes() for y in UOp.sink(*[x.uop for x in nn.state.get_parameters(model)]).toposort() if y.op is Ops.BUFFER]
  print(f"using model \"{model_name}\" with {sum(file_sizes):,} bytes and {sum(x.numel() for x in nn.state.get_parameters(model)):,} params")

  # get tokenizer
  tok = SimpleTokenizer.from_gguf_kv(kv)

  # warmup the JIT
  if args.warmup or args.serve:
    # run 2 tokens through the model twice to capture the JIT before serving
    with Context(DEBUG=max(DEBUG.value, 1)):
      for _ in range(2): list(zip(range(2), model.generate([0])))

  # start server
  if args.serve:
    vis = Qwen3VLVis(size="2B")
    image = cv2.cvtColor(cv2.imread("images/micra.jpg"), cv2.COLOR_BGR2RGB)
    prefill_img(vis=vis, lang=model, image=image, start_pos=len(model._cached_tokens), res=(640, 640))
    tokens = [0] * (((640 * 640) // (32*32)) + 8) # will this work lol
    model._cached_tokens.extend(tokens)
    LLMServer(('', args.serve), model, model_name, tok).serve_forever()

  # do benchmark
  if args.benchmark is not None:
    gen = model.generate(toks:=[tok.bos_id or 0])
    for i in range(args.benchmark):
      profile_marker(f"decode @ {i}")
      GlobalCounters.reset()
      with Timing(on_exit=lambda x: f", {1e9/x:6.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s,"
                  f" {GlobalCounters.global_mem//1000000}/{GlobalCounters.mem_used//1000000} MB  --  "+\
                  tok.decode(toks).replace("\n", "\\n")): next(gen)
    exit(0)

  # interactive chat
  ids: list[int] = tok.prefix()
  while 1:
    try:
      ids += tok.role("user") + tok.encode(input('>>> ')) + tok.end_turn() + tok.role("assistant")
    except EOFError:
      break
    dec = tok.stream_decoder()
    for next_id in model.generate(ids):
      sys.stdout.write(dec(next_id) if not tok.is_end(next_id) else dec() + "\n\n")
      sys.stdout.flush()
      if tok.is_end(next_id): break

if __name__ == "__main__": main()
