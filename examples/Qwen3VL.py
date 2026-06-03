from tinygrad import Tensor, TinyJit, dtypes, nn
from tinygrad.llm.gguf import gguf_load
from tinygrad.helpers import fetch
from tinygrad.nn.state import load_state_dict
import cv2 # todo resize in UI instead?
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