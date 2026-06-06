from tinygrad import Tensor, TinyJit, dtypes, nn
from tinygrad.llm.gguf import gguf_load
from tinygrad.helpers import fetch
from tinygrad.nn.state import load_state_dict
import cv2 # todo resize in UI instead?

#https://github.com/huggingface/transformers/blob/1316cd76c0ce328228e08d55dc257484961b074c/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L129
def rotate_half(x):
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  ret = Tensor.cat(-x2, x1, dim=-1)
  return ret

def apply_rotary_pos_emb_vision(query, key, cos, sin): return (query * cos) + (rotate_half(query) * sin), (key * cos) + (rotate_half(key) * sin)
  
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
  patches = (image - 127.5) / 127.5 # todo use mean and std
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
      .expand(-1, -1, -1, -1, -1, -1, vis.merge_size, -1, -1)
      .reshape(
          batch_size,
          grid_h * grid_w,
          channel * vis.merge_size * vis.patch_size * vis.patch_size,
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

def meshgrid(x, y):
  grid_x = Tensor.cat(*[x[idx:idx+1].expand(y.shape).unsqueeze(0) for idx in range(x.shape[0])])
  grid_y = Tensor.cat(*[y.unsqueeze(0)]*x.shape[0])
  return grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)

def get_vision_bilinear_indices_and_weights(h: int, w: int, num_grid_per_side: int, merge_size: int ) -> tuple[Tensor, Tensor]:
  side = num_grid_per_side

  h_grid = Tensor.linspace(0, side - 1, h)
  w_grid = Tensor.linspace(0, side - 1, w)
  h_floor = h_grid.cast(dtypes.int)
  w_floor = w_grid.cast(dtypes.int)

  h_ceil = (h_floor + 1).clamp(max_=side - 1)
  w_ceil = (w_floor + 1).clamp(max_=side - 1)

  h_frac = h_grid - h_floor
  w_frac = w_grid - w_floor

  h_floor_offset = h_floor * side
  h_ceil_offset = h_ceil * side

  corner_indices = Tensor.stack(
    (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
    (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
    (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
    (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
  )
  corner_weights = Tensor.stack(
    ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
    ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
    (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
    (h_frac[:, None] * w_frac[None, :]).flatten(),
  )

  h_idx = Tensor.arange(h).view(h // merge_size, merge_size)
  w_idx = Tensor.arange(w).view(w // merge_size, merge_size)
  reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).transpose(1, 2).flatten()
  bilinear_indices = corner_indices[:, reorder].reshape(4, -1)
  bilinear_weights = corner_weights[:, reorder].reshape(4, -1)
  return bilinear_indices, bilinear_weights

def get_vision_position_ids(h: int, w:int, merge_size: int):
  hpos_ids = Tensor.arange(h).unsqueeze(1).expand(-1, w)
  hpos_ids = hpos_ids.reshape(h // merge_size, merge_size, w // merge_size, merge_size).transpose(1, 2).flatten()
  wpos_ids = Tensor.arange(w).unsqueeze(0).expand(h, -1)
  wpos_ids = wpos_ids.reshape(h // merge_size, merge_size, w // merge_size, merge_size).transpose(1, 2).flatten()
  pos_ids = Tensor.stack(hpos_ids, wpos_ids, dim=-1).repeat(1, 1)
  return pos_ids

class Qwen3VLVis():
  def __init__(self, size="2B"):
    kv, state_dict = gguf_load(fetch(f"https://huggingface.co/Qwen/Qwen3-VL-{size}-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-{size}-Instruct-F16.gguf"))
    self.merge_size = kv["clip.vision.spatial_merge_size"]
    self.patch_size = kv["clip.vision.patch_size"]
    self.v = Qwen3VisBlocks(kv=kv, weights=state_dict)
    self.mm = [nn.Linear(*state_dict["mm.0.weight"].shape[::-1], bias=True), None, nn.Linear(*state_dict["mm.2.weight"].shape[::-1], bias=True)]
    state_dict["v.patch_embd.weight1"] = state_dict["v.patch_embd.weight.1"]
    load_state_dict(self, state_dict)
    #https://github.com/huggingface/transformers/blob/15bb519bd4277f4ab5309154aedf3c231e8b4ca8/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L98
    self.inv_freq = 1.0 / (10000.0 ** (Tensor.arange(0, 32, 2, dtype=dtypes.float) / 32))

  # https://github.com/huggingface/transformers/blob/15bb519bd4277f4ab5309154aedf3c231e8b4ca8/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L679
  def __call__(self, pixel_values, image_grid_size):
    grid_hs, grid_ws = image_grid_size    
    idx_tensor, weight_tensor = get_vision_bilinear_indices_and_weights(h=grid_hs, w=grid_ws, num_grid_per_side=self.v.num_grid_per_side, merge_size=self.merge_size)
    pos_ids = get_vision_position_ids(h=grid_hs, w=grid_ws, merge_size=self.merge_size)

    pos_embeds = (self.v.position_embd(idx_tensor) * weight_tensor[:, :, None]).sum(axis=0)

    w = Tensor.stack(self.v.patch_embd.weight, self.v.patch_embd.weight1, dim=2)
    w = w.reshape(w.shape[0], w.shape[1] * w.shape[2], w.shape[3], w.shape[4])
    hidden_states = pixel_values.reshape(-1, *w.shape[1:])
    hidden_states = hidden_states.conv2d(weight=w, bias=self.v.patch_embd.bias, stride=(self.patch_size, self.patch_size), padding=(0, 0), dilation=(1, 1), groups=1)
    hidden_states = hidden_states.view(hidden_states.shape[0], -1)
    hidden_states += pos_embeds

    rotary_pos_emb = (pos_ids.unsqueeze(-1) * self.inv_freq).flatten(1)
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

class Qwen3PatchEmbed:
  def __init__(self, kv=None, weights=None):
    self.weight = Tensor.zeros(weights["v.patch_embd.weight"].shape)
    self.weight1 = Tensor.zeros(weights["v.patch_embd.weight.1"].shape)
    self.bias = Tensor.zeros(kv["clip.vision.embedding_length"])
    
class Qwen3VisBlocks:
  def __init__(self, kv=None, weights=None):
    self.blk = []
    for _ in range(kv["clip.vision.block_count"]): self.blk.append(Qwen3VisBlock(kv, weights=weights))
    self.patch_embd = Qwen3PatchEmbed(kv=kv, weights=weights)
    #https://github.com/huggingface/transformers/blob/effde20942e3f82a1b97449f60b3a48c5ff96145/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L628
    self.num_grid_per_side = int(weights["v.position_embd.weight"].shape[0]**0.5)
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

class Qwen3VisBlock:
  def __init__(self, kv=None, weights=None):
    self.num_heads = kv["clip.vision.attention.head_count"]
    self.ffn_up = nn.Linear(kv["clip.vision.embedding_length"], kv["clip.vision.feed_forward_length"])
    self.ffn_down = nn.Linear(kv["clip.vision.feed_forward_length"], kv["clip.vision.embedding_length"])
    self.ln1 = nn.LayerNorm(kv["clip.vision.embedding_length"], eps=1e-6, elementwise_affine=True)
    self.ln2 = nn.LayerNorm(kv["clip.vision.embedding_length"], eps=1e-6, elementwise_affine=True)
    self.attn_out = nn.Linear(kv["clip.vision.embedding_length"], kv["clip.vision.embedding_length"])
    self.attn_qkv = nn.Linear(*weights["v.blk.0.attn_qkv.weight"].shape[::-1])
  
  def __call__(self, hidden_states, cos, sin):
    hidden_states_input = self.ln1(hidden_states)
    qkv = self.attn_qkv(hidden_states_input)
    # https://github.com/huggingface/transformers/blob/1316cd76c0ce328228e08d55dc257484961b074c/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L186
    qkv = qkv.reshape(qkv.shape[0], 3, self.num_heads, -1).permute(1, 0, 2, 3)
    query, key, value = qkv[0], qkv[1], qkv[2]
    query, key = apply_rotary_pos_emb_vision(query, key, cos, sin)
    query = query.transpose(0, 1).unsqueeze(0)
    value = value.transpose(0, 1).unsqueeze(0)
    key = key.transpose(0, 1).unsqueeze(0)

    attn_output = query.scaled_dot_product_attention(key, value)

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(attn_output.shape[1], -1)
    attn_output = self.attn_out(attn_output)

    hidden_states += attn_output
    norm = self.ln2(hidden_states)
    norm = self.ffn_up(norm).gelu()
    norm = self.ffn_down(norm)
    return hidden_states + norm