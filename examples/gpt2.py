#!/usr/bin/env python3
"""
GPT-2 模型实现 — 基于 tinygrad 深度学习框架
==============================================

本文件实现了完整的 GPT-2 语言模型，包括：
  1. 多头自注意力机制（Multi-Head Self-Attention）with KV Cache
  2. 前馈网络（FeedForward / MLP）
  3. Transformer Block（Pre-Norm 结构）
  4. 完整的 Transformer 模型
  5. 自回归文本生成（Autoregressive Generation）

tinygrad 技术要点（贯穿整个文件）：
  - 惰性求值（Lazy Evaluation）：Tensor 操作只构建计算图，不立即执行
  - TinyJit（JIT 编译）：将计算图编译为优化的底层 kernel，缓存复用
  - Variable（符号变量）：JIT 编译时的占位符，支持动态形状特化
  - UOp（微操作）：tinygrad 的中间表示（IR），用于表示和优化计算
  - realize()：触发实际的内存分配和 kernel 执行
  - schedule + codegen 管线：从高层 op → 调度（schedule）→ 底层代码生成
"""

import os, argparse, contextlib
from typing import Optional, Union

# tiktoken: OpenAI 开源的 BPE 分词器，将文本转换为 token ID 序列
with contextlib.suppress(ImportError): import tiktoken

# === tinygrad 核心导入 ===
# Tensor:     核心张量类，所有操作都是惰性的（构建计算图，不立即执行）
#             Tensor 不存储实际数据，只记录操作链（LazyBuffer）
# TinyJit:    JIT 编译器，将计算图编译为底层 kernel（如 CUDA/Metal/GPU）
#             首次调用时编译并缓存，后续相同形状的调用直接复用
# Device:     默认计算设备（通过 DEFAULT 环境变量或自动检测）
# GlobalCounters: 全局性能计数器（运算量、内存带宽等）
# Variable:   符号变量，用于 JIT 编译时的输入特化
#             例如 Variable("start_pos", 0, 1023) 表示一个范围为 [0,1023] 的符号值
#             JIT 会为每个实际绑定的值生成特化的 kernel
# dtypes:     数据类型定义（float32, float16 等）
from tinygrad import Tensor, TinyJit, Device, GlobalCounters, Variable, dtypes

# UOp: tinygrad 的底层中间表示（IR），是所有计算的基础单位
#      例如加法、乘法、reshape 等都是 UOp
#      Tensor 的高层操作最终都会转换为 UOp 图，然后经过 schedule → codegen 生成 kernel
from tinygrad.uop.ops import UOp

from tinygrad.helpers import Timing, DEBUG, JIT, getenv, fetch, colored, trange
from tinygrad.llm.gguf import gguf_load
from tinygrad.nn import Embedding, Linear, LayerNorm
from tinygrad.nn.state import torch_load, load_state_dict, get_state_dict
from extra.bench_log import BenchEvent, WallTimeEvent

MAX_CONTEXT = getenv("MAX_CONTEXT", 128)  # 最大上下文长度（prompt + 生成 token 的总上限）
HALF = getenv("HALF")  # 是否启用 FP16（半精度）推理，可减少显存占用并加速


# =============================================================================
# Multi-Head Self-Attention（多头自注意力机制）
# =============================================================================
class Attention:
  """
  多头自注意力 + KV Cache

  技术原理：
    自注意力公式：Attention(Q,K,V) = softmax(Q·K^T / √d) · V

    其中：
      - Q (Query):  当前 token 想要"查询"什么信息
      - K (Key):    每个 token "拥有"什么信息的索引
      - V (Value):  每个 token 实际存储的信息内容
      - √d:         缩放因子，防止点积过大导致 softmax 梯度消失

    多头机制：
      将 d_model 维度的 Q/K/V 拆分为 n_heads 个独立的 head
      每个 head 关注不同的信息维度，最后拼接起来
      例如 dim=768, n_heads=12 → 每个 head_dim=64

    KV Cache（生成推理的关键优化）：
      自回归生成时，每步只产生一个新 token
      如果每次都重新计算整个序列的 K/V，会造成大量的重复计算
      KV Cache 将之前 token 的 K/V 缓存下来，新 token 只需：
        1. 计算新 token 的 Q
        2. 用新 token 的 K/V 更新缓存
        3. Q 只与缓存中的 K/V 做 attention
      计算复杂度从 O(n²) 降到 O(n)

    tinygrad 视角：
      - Tensor.zeros 创建 KV Cache 时并不实际分配内存（惰性求值）
      - .contiguous().realize() 才触发实际的内存分配和初始化
      - .assign() 是原地更新操作，对应底层的内存拷贝 kernel
      - .shrink() 是零拷贝的视图（view）操作，只改变元数据不拷贝数据
  """
  def __init__(self, dim, n_heads):
    # c_attn: 将输入映射到 Q、K、V 三组向量（合并为一次线性变换以提高效率）
    #         输出维度 3*dim，在前向传播时再拆分成 Q/K/V
    self.c_attn = Linear(dim, 3*dim, bias=True)

    # c_proj: 将多头注意力的拼接结果投影回原始维度
    self.c_proj = Linear(dim, dim, bias=True)

    self.n_heads = n_heads
    self.dim = dim
    self.head_dim = dim // n_heads  # 每个 attention head 的维度

  def __call__(self, x:Tensor, start_pos:Variable, mask:Optional[Tensor]) -> Tensor:
    # --- JIT 辨析 ---
    # 在消费 prompt 阶段（seqlen > 1）或有 mask 时，不使用符号形状
    # 因为在实际的 shape 已知情况下，直接展开计算更高效
    # tinygrad 的 Variable 被 .val 取出后变为具体整数，后续操作可用具体 shape
    if mask is not None or start_pos.val == 0:
      # no symbolic shape qkv when consuming prompts
      start_pos = start_pos.val

    # 可选半精度转换，减少计算量和内存
    if HALF: x = x.half()

    # --- Q/K/V 投影 ---
    # c_attn(x) 输出 shape: (batch, seqlen, 3*dim)
    # reshape 为 (batch, seqlen, 3, n_heads, head_dim)
    # 然后沿第 2 维（"3" 维度）拆分为 Q、K、V
    # 每个都是 (batch, seqlen, n_heads, head_dim)
    #
    # tinygrad 视角：
    #   reshape 是零拷贝操作，只改变 Tensor 的元数据（shape/stride）
    #   真正的计算推迟到后续的 matmul/softmax 等操作
    xqkv = self.c_attn(x).reshape(None, None, 3, self.n_heads, self.head_dim)
    xq, xk, xv = [xqkv[:, :, i, :, :] for i in range(3)]
    bsz, seqlen, _, _ = xq.shape

    # --- KV Cache 初始化和管理 ---
    # 首次调用时创建 KV Cache，shape: (2, batch, MAX_CONTEXT, n_heads, head_dim)
    #   第 0 维: 2 表示 Key 和 Value
    #
    # tinygrad 视角（关键！）：
    #   Tensor.zeros(...)        — 惰性操作，只记录"需要创建零张量"，不分配内存
    #   .contiguous()            — 确保内存连续布局（合并可能的非连续 stride）
    #   .realize()               — 触发实际执行！此刻才分配 GPU 内存并填充零值
    #
    #   为什么必须 realize()：
    #     后续 .assign() 需要原地写入 KV Cache。如果 Cache 尚未实体化（materialized），
    #     则无法在原地写入。realize() 将惰性图转换为实际的内存块。
    if not hasattr(self, "cache_kv"):
      self.cache_kv = Tensor.zeros(2, bsz, MAX_CONTEXT, self.n_heads, self.head_dim, dtype=x.dtype).contiguous().realize()

    # --- 更新 KV Cache ---
    # 将当前 token 的 K 和 V 存入 Cache 的对应位置
    #
    # tinygrad 视角：
    #   Tensor.stack(xk, xv)    — 将 K 和 V 沿新维度堆叠，shape 变为 (2, batch, seqlen, n_heads, head_dim)
    #   .assign(...)             — 原地赋值！将 stack 结果拷贝到 Cache 的指定位置
    #   这是一个写操作（store kernel），生成对应的底层 memory copy 代码
    #
    #   start_pos:start_pos+seqlen 是 Cache 中的写入位置
    #   对于 prompt 阶段：seqlen 可能 > 1，一次写入多个 token 的 KV
    #   对于生成阶段：seqlen = 1，每次只写入一个 token 的 KV
    self.cache_kv[:, :, start_pos:start_pos+seqlen, :, :].assign(Tensor.stack(xk, xv)).realize()

    # --- 从 Cache 读取 ---
    # 如果已有历史 token（start_pos > 0），则从 Cache 读取完整的 K/V 序列
    # 否则（第一步），直接使用当前计算的 xk、xv
    if start_pos > 0:
      keys = self.cache_kv[0][:, :start_pos+seqlen, :, :]
      values = self.cache_kv[1][:, :start_pos+seqlen, :, :]
    else:
      keys = xk
      values = xv

    # --- Scaled Dot-Product Attention ---
    # transpose: (batch, seqlen, n_heads, head_dim) → (batch, n_heads, seqlen, head_dim)
    # 将 head 维度提前，便于并行计算各 head 的 attention
    #
    # tinygrad 视角：
    #   scaled_dot_product_attention 是一个融合操作（fused op）：
    #     内部包含 Q·K^T（matmul）、缩放、可选 mask、softmax、×V（matmul）
    #   在 schedule 阶段，tinygrad 可能会将这些操作融合为一个高效的 kernel
    #   （例如使用 FlashAttention 算法来减少 HBM 读写）
    xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
    return self.c_proj(xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2).reshape(bsz, seqlen, self.dim))


# =============================================================================
# Feed-Forward Network（前馈网络 / MLP）
# =============================================================================
class FeedForward:
  """
  双层全连接网络，使用 GELU 激活函数

  结构：input → Linear(dim → 4*dim) → GELU → Linear(4*dim → dim) → output

  技术原理：
    Transformer 中的 FFN 使用"扩维-激活-缩维"的设计：
      - 先扩充到 4 倍维度，增加模型容量和表达能力
      - GELU（Gaussian Error Linear Unit）是 Transformer 中常用的激活函数
        相比 ReLU，GELU 更平滑，在负值区域保留部分信息
        GELU(x) ≈ x · Φ(x)，其中 Φ 是标准正态分布的 CDF

  tinygrad 视角：
    self.c_fc(x).gelu() 构建一个包含 matmul + gelu 的计算图
    由于惰性求值，此时没有任何计算发生
    后续的 .realize() 才会触发 kernel 生成和执行
  """
  def __init__(self, dim, hidden_dim):
    # c_fc: "Fully Connected"，将维度从 dim 扩展到 hidden_dim（= 4*dim）
    self.c_fc = Linear(dim, hidden_dim, bias=True)
    # c_proj: 将维度从 hidden_dim 投影回 dim
    self.c_proj = Linear(hidden_dim, dim, bias=True)

  def __call__(self, x:Tensor) -> Tensor:
    # c_fc(x) → gelu → c_proj
    # 扩展 → 非线性激活 → 压缩回原始维度
    return self.c_proj(self.c_fc(x).gelu())


# =============================================================================
# Transformer Block（Transformer 基本构建块）
# =============================================================================
class TransformerBlock:
  """
  Pre-Norm Transformer Block

  结构（Pre-Norm 设计，即 LayerNorm 在子层之前）：
    x → LayerNorm → Attention → + (残差连接)
      → LayerNorm → FeedForward → + (残差连接)

  技术原理：
    残差连接（Residual Connection）：
      输出 = 输入 + 子层(输入)
      解决深层网络的梯度消失问题，使 Transformer 可以堆叠几十层

    Pre-Norm vs Post-Norm：
      原始 Transformer（Post-Norm）：子层 → LayerNorm → + 残差
      GPT-2 使用 Pre-Norm：LayerNorm → 子层 → + 残差
      Pre-Norm 训练更稳定，梯度流动更顺畅

  tinygrad 视角：
    每个 Block 内的所有操作（norm、attention、ffn、加法）都只是构建计算图
    .contiguous() 确保输出张量在内存中连续排列
    实际的内存在父级调用 .realize() 时才分配
  """
  def __init__(self, dim, n_heads, norm_eps):
    self.attn = Attention(dim, n_heads)
    self.mlp = FeedForward(dim, 4*dim)
    self.ln_1 = LayerNorm(dim, norm_eps)  # Attention 之前的 LayerNorm
    self.ln_2 = LayerNorm(dim, norm_eps)  # FeedForward 之前的 LayerNorm

  def __call__(self, x:Tensor, start_pos:Variable, mask:Optional[Tensor]):
    # 第一个子层：Pre-Norm → Attention → 残差连接
    # .float() 确保 attention 输出转回 float32（如果使用了半精度）
    h = x + self.attn(self.ln_1(x), start_pos, mask).float()
    # 第二个子层：Pre-Norm → FeedForward → 残差连接
    # .contiguous() 确保内存连续布局
    return (h + self.mlp(self.ln_2(h))).contiguous()


# =============================================================================
# Transformer 主模型
# =============================================================================
class Transformer:
  """
  完整的 GPT-2 Transformer 模型

  结构：
    Token Embedding + Position Embedding → [TransformerBlock × n_layers] → LayerNorm → LM Head

  tinygrad JIT 编译详解（本文件最核心的技术概念）：

    1. 惰性求值（Lazy Evaluation）：
       每个 Tensor 操作（matmul、add、softmax...）不会立即在 GPU 上执行
       而是在内存中构建一个 LazyBuffer 树，记录所有操作
       只有当调用 .realize() 时，整个计算图才被：
         a. schedule（调度：将高层 op 拆解为底层 UOp）
         b. 优化（融合相邻 kernel、减少内存分配）
         c. codegen（生成目标后端代码：CUDA/Metal/GPU/LLVM）
         d. 执行

    2. TinyJit：
       首次调用 model(tokens, start_pos) 时：
         - 执行完整的 schedule + codegen 流程
         - 将生成的 kernel 与输入特征（shape、dtype、Variable 绑定值）缓存
       后续相同特征的调用：
         - 跳过 schedule + codegen，直接复用已编译的 kernel
         - 大幅降低延迟（schedule + codegen 可能耗时数百 ms）

    3. Variable（符号变量）：
       Variable("start_pos", 1, 1023).bind(42) 表示：
         - 创建一个类型为 "start_pos"、范围 [1, 1023] 的符号变量
         - 实际运行时绑定值为 42
       JIT 会根据 bind 的值进行特化（specialize）：
         - 例如 start_pos=0 和 start_pos=42 会生成不同的 kernel
         - 但 start_pos=42 和 start_pos=43 可能共享同一个 kernel（如果形状相同）

    4. UOp / Schedule / Codegen 管线：
       Tensor 高层操作（matmul、softmax...）
         ↓
       LazyBuffer（惰性缓冲区，记录操作图）
         ↓ .realize() 触发
       Schedule（将 LazyBuffer 图转换为 ScheduleItem 列表）
         - 融合相邻的 element-wise 操作
         - 将 matmul 等操作分配给特定的优化 kernel（如中使用的加速库或手写 kernel）
         ↓
       Lower（将 ScheduleItem 转换为 UOp 图）
         - UOp 是底层 IR，直接对应循环、加载/存储、算术等
         ↓
       Codegen（从 UOp 图生成后端代码）
         - GPU: 生成对应的着色器代码（如 WGSL/MSL/CUDA）
         - CPU: 生成 LLVM IR → 机器码
         ↓
       Execution（在目标设备上执行 kernel）
  """
  def __init__(self, dim, n_heads, n_layers, norm_eps, vocab_size, max_seq_len=1024):
    self.vocab_size = vocab_size

    # Token Embedding: 将 token ID（整数）映射为稠密向量
    # 例如 vocab_size=50257, dim=768 → Embedding 矩阵 shape (50257, 768)
    self.wte = Embedding(vocab_size, dim)

    # Position Embedding: 将位置索引（整数）映射为稠密向量
    # 使模型能够感知 token 在序列中的位置
    # 例如 max_seq_len=1024, dim=768 → Embedding 矩阵 shape (1024, 768)
    self.wpe = Embedding(max_seq_len, dim)

    # Transformer Blocks: 堆叠 n_layers 层
    self.h = [TransformerBlock(dim, n_heads, norm_eps) for _ in range(n_layers)]

    # 最终 LayerNorm（Pre-Norm 架构中的最终归一化）
    self.ln_f = LayerNorm(dim, norm_eps)

    # LM Head: 将最后一个 hidden state 映射到词表维度的 logits
    # 在 GPT-2 中，lm_head.weight 与 wte.weight 是共享的（Weight Tying）
    self.lm_head = Linear(dim, vocab_size, bias=False)

    # === TinyJit 初始化 ===
    # TinyJit(self.forward) 包装 forward 方法
    # 首次调用时编译并缓存 kernel
    # 后续调用直接复用，只执行 kernel 不重复编译
    self.forward_jit = TinyJit(self.forward)

  def forward(self, tokens:Union[Tensor,UOp], start_pos:Variable, temperature:float=0.0):
    """
    前向传播

    参数：
      tokens:     输入 token（Tensor 表示 prompt 阶段，UOp 表示单个 token 生成阶段）
      start_pos:  序列起始位置（0 表示从开头开始）
      temperature: 采样温度（0 表示贪心解码）

    Tensor vs UOp 的区别（tinygrad 内部机制）：
      - Tensor:  完整的张量数据，用于 prompt 处理（一次输入多个 token）
      - UOp:     符号表示，用于逐 token 生成（每次输入 1 个 token 的符号索引）
                 使用 UOp 可以在 JIT 图中保留符号信息，便于优化
    """
    # allpos: 位置索引 [0, 1, 2, ..., MAX_CONTEXT-1]
    # 在 Position Embedding 查询时使用
    # .realize() 将其立即实体化（因为值固定，无需惰性）
    if not hasattr(self, 'allpos'): self.allpos = Tensor.arange(0, MAX_CONTEXT).reshape(1, -1).realize()

    # --- Token Embedding ---
    if isinstance(tokens, UOp):
      # 单个 token 生成模式：tokens 是一个 UOp 符号，shrink 用于在 Embedding 矩阵中"选取"对应行
      # shrink 是零拷贝的视图操作
      seqlen = 1
      tok_emb = self.wte.weight.shrink(((tokens, tokens+1), None))
    else:
      # Prompt 处理模式：tokens 是一个完整的 Tensor，使用 Embedding 查表
      seqlen = tokens.shape[1]
      tok_emb = self.wte(tokens)

    # --- Position Embedding ---
    # 根据序列位置查找位置 Embedding
    # 当 start_pos.val == 0 时（第一步），需要查全部 seqlen 个位置
    # 否则（后续步骤），只需要查最新一个位置
    selected_pos = (0, seqlen) if start_pos.val == 0 else (start_pos, start_pos+1)
    pos_emb = self.wpe(self.allpos.shrink((None, selected_pos)))

    # Token Embedding + Position Embedding 相加得到输入表示
    h = tok_emb + pos_emb

    # 可选半精度
    if HALF: h = h.half()

    # --- Causal Mask（因果遮罩）---
    # 保证每个 token 只能看到它自己及其之前的 token（不能"偷看"未来信息）
    # 仅在 prompt 处理（seqlen > 1）时需要 mask
    # 生成单个 token 时不需要（因为只关注自己 + 之前的缓存）
    #
    # triu(start_pos+1): 上三角矩阵，对角线偏移 start_pos+1
    # 确保位置 i 只能关注位置 [0, i]（对于 i < start_pos+seqlen）
    mask = Tensor.full((1, 1, seqlen, start_pos.val+seqlen), float("-inf"), dtype=h.dtype).triu(start_pos.val+1) if seqlen > 1 else None

    # --- 逐层前向传播 ---
    for hi in self.h: h = hi(h, start_pos, mask)

    # --- LM Head ---
    # LayerNorm → Linear(dim → vocab_size) 得到 logits
    logits = self.lm_head(self.ln_f(h))

    # --- 边界情况：空 prompt ---
    if logits.shape[1] == 0:
      # special case for empty prompt
      logits = Tensor.ones((logits.shape[0], self.vocab_size), dtype=logits.dtype, device=logits.device)
    else:
      # 只取最后一个位置的 logits（自回归生成只需要预测下一个 token）
      logits = logits[:, -1, :]

    # --- 采样 ---
    # temperature < 1e-6: 贪心解码（argmax），选择概率最高的 token
    # temperature > 0:    随机采样，temperature 越高越随机
    if temperature < 1e-6:
      ret = logits.argmax(-1)
    else:
      ret = (logits / temperature).softmax().multinomial()
    # .flatten().realize() 触发整个计算图的执行！
    return ret.flatten().realize()

  def __call__(self, tokens:Union[Tensor,UOp], start_pos:Variable, temperature:float=0.0) -> Tensor:
    """
    模型调用入口

    JIT 策略（自动选择是否使用 JIT 编译）：
      - 使用 forward_jit（缓存编译结果）当：
        1. JIT 环境变量已启用
        2. tokens 是 UOp 类型（单 token 生成，可复用 kernel）
        或 tokens 是单 token 的 Tensor（seqlen=1，也可复用）
      - 使用 forward（每次重新编译）当：
        处理多 token prompt（seqlen > 1），形状每次不同，不值得缓存

    为什么逐 token 生成适合 JIT：
      每步的输入形状完全相同（都是 1 个 token），JIT 缓存命中率极高
      这使 tinygrad 的生成速度接近原生框架

    为什么 prompt 阶段不适合 JIT：
      prompt 长度可变，每个 prompt 可能有不同的 seqlen
      如果每次都缓存，会导致缓存膨胀，而且命中率低
    """
    forward = (self.forward_jit if JIT and (isinstance(tokens, UOp) or tokens.shape[1] == 1) else self.forward)
    return forward(tokens, start_pos, temperature)


# =============================================================================
# GPT-2 模型参数配置
# =============================================================================
VOCAB_SIZE = 50257  # GPT-2 词表大小
MODEL_PARAMS = {
  # n_layers: Transformer Block 层数
  # n_heads:  注意力头数
  # dim:      隐藏层维度（d_model）
  # norm_eps: LayerNorm 的 epsilon（防止除零）
  'gpt2':         dict(n_layers=12, n_heads=12, dim=768,  norm_eps=1e-5, vocab_size=VOCAB_SIZE),   # 124M 参数
  'gpt2-medium':  dict(n_layers=24, n_heads=16, dim=1024, norm_eps=1e-5, vocab_size=VOCAB_SIZE),  # 350M 参数
  'gpt2-large':   dict(n_layers=36, n_heads=20, dim=1280, norm_eps=1e-5, vocab_size=VOCAB_SIZE),  # 774M 参数
  'gpt2-xl':      dict(n_layers=48, n_heads=25, dim=1600, norm_eps=1e-5, vocab_size=VOCAB_SIZE),  # 1558M 参数
}


# =============================================================================
# GPT-2 模型加载和文本生成
# =============================================================================
class GPT2:
  @staticmethod
  def build(model_size="gpt2"):
    """
    从 HuggingFace 下载 GPT-2 权重并构建模型

    流程：
      1. 初始化 tokenizer（tiktoken BPE 分词器）
      2. 创建 Transformer 模型结构（随机初始化权重）
      3. 下载预训练权重（PyTorch .bin 格式）
      4. 将权重加载到模型中

    权重转置（Conv1D → Linear）：
      HuggingFace 的 GPT-2 原始实现使用 Conv1D（1D 卷积），
      其权重矩阵与标准 Linear 层的结果互为转置
      因此在加载时需要将特定权重的 .T（转置）传递给 Linear

    Weight Tying（权重共享）：
      GPT-2 中 lm_head（输出投影层）和 wte（token embedding）共享权重
      这减少了参数量（约 50257 × 768 ≈ 38M 参数），且有助于训练

    tinygrad 视角：
      load_state_dict 将 NumPy 数组赋值给对应的 Tensor 参数
      HALF 模式下：.half() 转换精度，.realize() 触发实际转换并释放原内存
    """
    tokenizer = tiktoken.get_encoding("gpt2")

    model = Transformer(**MODEL_PARAMS[model_size])
    # fetch: 从 HuggingFace 下载模型权重文件，缓存到本地
    # torch_load: 读取 PyTorch 的 .bin 文件，返回 dict[str, Tensor]
    weights = torch_load(fetch(f'https://huggingface.co/{model_size}/resolve/main/pytorch_model.bin'))
    # special treatment for the Conv1D weights we need to transpose
    # HuggingFace 原始 GPT-2 使用 Conv1D 实现，其权重矩阵需要转置才能用于 Linear
    transposed = ('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')
    for k in weights:
      if k.endswith(transposed):
        weights[k] = weights[k].T
    # lm head and wte are tied
    # 输出投影层与 Token Embedding 共享权重
    weights['lm_head.weight'] = weights['wte.weight']

    with WallTimeEvent(BenchEvent.LOAD_WEIGHTS):
      load_state_dict(model, weights)

      # 如果启用半精度：将所有参数转换为 float16
      # .replace() 是原地替换（重用同一个 Tensor 对象名）
      if HALF:
        for l in get_state_dict(model).values():
          l.replace(l.half().realize())

    return GPT2(model, tokenizer)

  @staticmethod
  def build_gguf(model_size: str):
    """
    从 GGUF 格式加载量化模型

    GGUF（GPT-Generated Unified Format）是 llama.cpp 项目的模型格式
    支持多种量化精度（Q4_0、Q5_1、Q8_0 等），可大幅减少模型体积和内存占用

    技术原理：
      量化（Quantization）：将浮点权重映射到低比特整数（如 4-bit、8-bit）
      例如 Q4_0: 每个权重 4.5 bits，模型体积缩小约 4 倍
      推理时反量化（dequantize）回浮点进行计算

    tinygrad 视角：
      Tensor.empty(..., device=f"disk:{fn}") → 创建一个映射到磁盘文件的张量
        tinygrad 支持 "disk" 设备，将文件内容直接映射为 Tensor（mmap）
        无需将整个文件读入内存，按需加载
      .to(Device.DEFAULT) → 将数据从磁盘传输到计算设备（GPU/CPU）
      gguf_load: 解析 GGUF 格式，提取元数据和权重张量
    """
    q_type = model_size[len("gpt2_gguf_"):].upper()
    fn = fetch(f"https://huggingface.co/PrunaAI/gpt2-GGUF-smashed/resolve/main/gpt2.{q_type}.gguf?download=true")
    # disk: 设备允许将磁盘文件映射为 Tensor（内存映射 I/O）
    gguf_tensor = Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}").to(Device.DEFAULT)
    kv_data, state_dict = gguf_load(gguf_tensor)

    # 从 GGUF 元数据中读取模型配置
    gpt2_params = {
      "dim": kv_data["gpt2.embedding_length"], "n_heads": kv_data["gpt2.attention.head_count"],
      "n_layers": kv_data["gpt2.block_count"], "norm_eps": kv_data["gpt2.attention.layer_norm_epsilon"],
      "vocab_size": VOCAB_SIZE, "max_seq_len": kv_data["gpt2.context_length"],
    }
    def _remap_gguf_key(key: str):
      """
      GGUF 格式的权重命名规范与 HuggingFace 不同，需要重映射
      例如 "blk.0.attn_qkv.weight" → "h.0.attn.c_attn.weight"
      """
      replaces = [
        ("blk.", "h."), (".attn_qkv.bias", ".attn.c_attn.bias"), (".attn_qkv.weight", ".attn.c_attn.weight"),
        (".ffn_norm.bias", ".ln_2.bias"), (".ffn_norm.weight", ".ln_2.weight"), (".attn_norm.bias", ".ln_1.bias"),
        (".attn_norm.weight", ".ln_1.weight"), (".attn_output.bias", ".attn.c_proj.bias"), (".attn_output.weight", ".attn.c_proj.weight"),
        (".ffn_up.bias", ".mlp.c_fc.bias"), (".ffn_up.weight", ".mlp.c_fc.weight"), (".ffn_down.bias", ".mlp.c_proj.bias"),
        (".ffn_down.weight", ".mlp.c_proj.weight"), ("token_embd.weight", "wte.weight"), ("output.weight", "lm_head.weight"),
        ("output_norm.bias", "ln_f.bias"), ("output_norm.weight", "ln_f.weight"), ("position_embd.weight", "wpe.weight"),
      ]
      for ostr, ns in replaces: key = key.replace(ostr, ns)
      return key
    state_dict = { _remap_gguf_key(k): v for k, v in state_dict.items() }
    model = Transformer(**gpt2_params)
    with WallTimeEvent(BenchEvent.LOAD_WEIGHTS):
      load_state_dict(model, state_dict)
    return GPT2(model, tiktoken.get_encoding("gpt2"))

  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer

  def generate(self, prompt:str, max_length:int, temperature:float, timing:bool=False, batch_size:int=1):
    """
    自回归文本生成（Autoregressive Generation）

    过程：
      1. Tokenize: 将 prompt 字符串转换为 token ID 列表（BPE 编码）
      2. Prefill（预填充）: 一次性将全部 prompt token 输入模型
         - 计算所有 prompt token 的 KV Cache
         - 得到第一个生成 token 的 logits
      3. Decode（逐 token 生成）:
         - 每次输入最新生成的 1 个 token
         - 利用 KV Cache 避免重复计算
         - 采样得到下一个 token
         - 重复直到达到 max_length 或遇到 EOS

    tinygrad JIT 的实际运作（结合 generate 方法）：
      ┌─────────────────────────────────────────────────────────┐
      │ 第 1 步（Prefill）：                                     │
      │   tokens = Tensor(prompt_tokens)  # shape (1, seqlen)    │
      │   model(tokens, start_pos=0)                             │
      │   → 不使用 JIT（seqlen > 1）                              │
      │   → 完整走 schedule → codegen → execute                  │
      │   → 计算所有 prompt token 的 KV Cache                    │
      ├─────────────────────────────────────────────────────────┤
      │ 第 2 步（首 token 生成）：                                │
      │   tokens = Variable("tokens", 0, VOCAB_SIZE-1).bind(tok) │
      │   model(tokens, start_pos=len(prompt))                   │
      │   → 使用 TinyJit！首次编译，缓存 kernel                  │
      │   → 只有 1 个 token，从 KV Cache 读取历史                 │
      ├─────────────────────────────────────────────────────────┤
      │ 第 3~N 步（后续 token 生成）：                            │
      │   tokens = Variable("tokens", ...).bind(tok)             │
      │   model(tokens, start_pos=...)                           │
      │   → JIT 缓存命中！直接复用已编译的 kernel                 │
      │   → 跳过 schedule + codegen，只执行 GPU kernel           │
      │   → 延迟极低（通常 < 1ms 的 kernel 启动开销）            │
      └─────────────────────────────────────────────────────────┘

    性能指标：
      - time_till_first: 从发送请求到收到第一个 token 的时间
                         包含 prefill 阶段（处理完整 prompt）
      - tokens_per_second: 生成阶段每秒产生的 token 数
      - GlobalCounters: tinygrad 自动记录的计算量和内存带宽
    """
    step_times = []
    prompt_tokens = self.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    toks = [prompt_tokens[:] for _ in range(batch_size)]  # 支持批处理（多个独立序列）
    start_pos = 0
    for _ in trange(max_length, disable=(timing==True)):
      GlobalCounters.reset()  # 重置性能计数器，开始统计当前 step
      if timing: print("")
      st = GlobalCounters.time_sum_s
      with Timing("ran model in ", on_exit=(lambda et: (f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on {Device.DEFAULT}" if DEBUG>=2 else "")+
                  f", {GlobalCounters.global_ops*1e-9:.2f} GOPS, {GlobalCounters.global_mem*1e-9:.2f} GB"+
                  (f", {GlobalCounters.global_mem*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s" if DEBUG>=2 else "")) if DEBUG else None, enabled=timing):
        with WallTimeEvent(BenchEvent.STEP):
          # --- 构造输入 ---
          if batch_size == 1 and len(toks[0][start_pos:]) == 1:
            # 单 token 生成：使用 Variable（JIT 优化路径）
            # Variable("tokens", 0, VOCAB_SIZE-1) 创建范围为 [0, VOCAB_SIZE-1] 的符号变量
            # .bind(tok) 将实际 token 值绑定到符号变量上
            # JIT 会为每个不同的 start_pos 值生成特化的 kernel
            tokens = Variable("tokens", 0, VOCAB_SIZE-1).bind(toks[0][start_pos])
          else:
            # 多 token（prompt 或批处理）：使用普通 Tensor
            tokens = Tensor([x[start_pos:] for x in toks])
          tok = self.model(tokens, Variable("start_pos", 1 if start_pos else 0, MAX_CONTEXT-1).bind(start_pos), temperature).tolist()
      step_times.append((GlobalCounters.time_sum_s-st)*1e3)
      start_pos = len(toks[0])  # 更新为当前序列总长度
      for i,t in enumerate(tok): toks[i].append(t)

    # 性能回归检测（CI 使用）
    if (assert_time:=getenv("ASSERT_MIN_STEP_TIME")):
      min_time = min(step_times)
      assert min_time < assert_time, f"Speed regression, expected min step time of < {assert_time} ms but took: {min_time} ms"
    return [self.tokenizer.decode(x) for x in toks]

# =============================================================================
# main — 命令行入口
# =============================================================================

if __name__ == "__main__":
  print(f"using {Device.DEFAULT} backend")
  default_prompt = "What is the answer to life, the universe, and everything?"

  parser = argparse.ArgumentParser(description='Run GPT2 in tinygrad', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--prompt', type=str, default=default_prompt, help="Phrase to start with")
  parser.add_argument('--count', type=int, default=100, help="Max number of tokens to generate")
  parser.add_argument('--temperature', type=float, default=0.8, help="Temperature in the softmax")
  parser.add_argument('--model_size', type=str, default="gpt2-medium", help="Size of model to use [gpt2, gpt2-medium, gpt2-large, gpt2-xl]")
  parser.add_argument('--timing', action='store_true', help="Print timing per token")
  parser.add_argument('--seed', type=int, help="Set the random seed")
  parser.add_argument('--batch_size', type=int, default=1, help="Set the input batch size")
  parser.add_argument('--benchmark', type=int, default=-1, help="Benchmark GPT with the given number of tokens")
  parser.add_argument('--noshow', action='store_true', help="Don't show the output")
  args = parser.parse_args()

  if args.seed is not None:
    Tensor.manual_seed(args.seed)

  print(f"using {args.model_size}")
  # 支持 HuggingFace 原始权重或 GGUF 量化权重
  gpt2 = GPT2.build_gguf(args.model_size) if args.model_size.startswith("gpt2_gguf_") else GPT2.build(args.model_size)

  if args.benchmark != -1:
    # Benchmark 模式：只运行单次前向传播，测量性能
    gpt2.model(Tensor.randint(args.batch_size, args.benchmark), Variable("a", 0, MAX_CONTEXT).bind(0)).realize()
  else:
    # 生成模式：自回归生成文本
    texts = gpt2.generate(args.prompt, args.count, args.temperature, timing=args.timing, batch_size=args.batch_size)
    if not args.noshow:
      print('Generating text...')
      if len(texts) == 1: print(texts[0])
      else:
        for i,text in enumerate(texts): print(colored(f"Response {i}:", "green"), text)

    # validate output!（回归测试：确保特定 prompt 的生成结果一致）
    if args.temperature == 0 and args.model_size == "gpt2-medium" and args.count == 10:
      expected = {
        default_prompt: "What is the answer to life, the universe, and everything?\n\nThe answer is that we are all one",
        "Hello.": "Hello. I'm a little late to the party, but",
      }
      try:
        assert texts[0] == expected[args.prompt]
        print(colored("output validated", "green"))
      except KeyError:
        pass
