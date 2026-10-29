#!/usr/bin/env python3
# K3 decode on one box: the trunk sharded over the gpus (tinygrad/llm/model.py), the routed experts streamed from the nvme disks (store.py)
# into a vram cache of slots per layer. at every moe layer each gpu writes its share of the router logits to host memory and waits
# (hostsignal/hostwait) while the host routes, reads the missing experts straight from the drives into vram and fills the slot tables.
# DEV=AMD:HIP PYTHONPATH=. python3 extra/k3/engine.py trunk.gguf "prompt"
import ctypes, functools, os, sys, time
os.environ.setdefault("BEAM", "2")  # the generic kernels searched (cached after the first launch): dry 47 -> 40.5 ms/token
from typing import Callable, cast
import numpy as np
from tinygrad import Tensor, Device, dtypes, nn
from tinygrad.device import Buffer, BufferSpec, MultiBuffer
from tinygrad.helpers import getenv, prod, round_up
from tinygrad.uop.ops import UOp
from tinygrad.llm import model as llm_model
from tinygrad.llm.model import Transformer, gated_activation
from tinygrad.llm.kernels.amd import Linear, expert_linear, partial_sum, p2p_allreduce, q8_act, QUANT_NAMES
from tinygrad.runtime.support.nvme import PRPList, OP_READ, OP_WRITE, PAGE
from extra.k3.store import open_store, host_io, read_manifest, write_manifest, STRIPE, EXPERTS
from extra.k3.draft import q8_0_linear

# an nvme submission entry (nvme.py SQE)
SQE = np.dtype([("cdw0", "<u4"), ("nsid", "<u4"), ("rsvd", "<u8"), ("mptr", "<u8"), ("prp1", "<u8"), ("prp2", "<u8"),
                ("cdw10", "<u4"), ("cdw11", "<u4"), ("cdw12", "<u4"), ("pad", "V12")])

# SLOTS: experts cached over all layers (6900 fill 4x32 GB, 6700 with the draft's 0.5 GB a gpu)
K, KMIN, SLOTS, HALF_LIFE = 16, 4, getenv("SLOTS", 6700 if getenv("DRAFT", "") else 6900), 8.0
# TAU: the keep-resident truncation of the routing, PREFETCH: the next layer's experts guessed per layer. ROWS: the score rows a layer
# writes, its router's and the guess of the next layer's
TAU, PREFETCH, ROWS = getenv("TAU", 0.03), getenv("PREFETCH", 5), 2
# TAUM: (with TAU) the share of a token's weight tau may take, the lightest first. RENORM=0: the kept keep their weights (not renormalized)
TAUM, RENORM = getenv("TAUM", 0.0), getenv("RENORM", 1)
# BETA: the routing's bonus of a cached expert (its key sigmoid + bias + BETA): one just out of the top k takes the place of one not cached
BETA = getenv("BETA", 0.0)
# TAUC: the experts under it left out cached or not (the gemvs they'd take)
TAUC = getenv("TAUC", 0.0)
# FILL: the experts a layer's tau skipped for not being cached read anyway (up to FILL a layer, after its guesses): cached next time
FILL = getenv("FILL", 0)
VICTIMS = 32  # slots picked for the next layer's misses while the gpu works
# K1: the experts of a layer in two parts, the first K1 of them in the order they're ready (cached first) and the rest: the gpu works on
# the first while the misses are read
K1 = getenv("K1", 8)
# KMAX: at most this many experts a token kept, the heaviest (the experts' gemvs are KMAX a token: a slot of weight 0 costs a real one)
KMAX = getenv("KMAX", K)
# DRY=1: no expert reads, the slots hold garbage. it times the gpu side and the host round trips
DRY = getenv("DRY", 0)
# CHUNKS: a prompt fed in chunks of these many tokens (the largest that fits first, multiples of 16), the rest (and its last token) a token a
# step. a step of more than CHUNK_MIN tokens is a chunk: all of every layer's experts read in, the gpus route it themselves (exact)
CHUNKS, CHUNK_MIN = [int(c) for c in getenv("CHUNKS", "1024,128").split(",") if c], 8
# STORE: the disk 0 offset of the store's manifest. 512 GiB: experts split evenly over the gpus (relayout.py), 0: the first 1024/512/512/1024
STORE = getenv("STORE", 512 << 30)

def host_tensor(dev:str, shape:tuple[int, ...], dtype) -> tuple[Tensor, np.ndarray]:
  """a buffer of dev in uncached host memory: the gpu reads and writes it over pcie, the host through the numpy view"""
  buf = Buffer(dev, prod(shape) * dtype.itemsize, options=BufferSpec(host=True, uncached=True, cpu_access=True)).ensure_allocated()
  host = buf.get_storage().host
  assert host is not None
  arr = np.frombuffer(host.mv.cast('B'), dtype=np.dtype(dtype.fmt), count=prod(shape)).reshape(shape)
  arr[:] = 0
  return Tensor(UOp.from_buffer(buf, dtype)).reshape(shape), arr

def vram_pages(buf:Buffer) -> list[int]:
  iface = Device[buf.device].iface  # type: ignore[attr-defined]
  bar, bar_size = iface.pci_dev.bar_info(iface.vram_bar)
  # a drive writing past the bar hits nothing: a completion timeout takes the machine down
  paddrs = buf.get_storage().meta.mapping.paddrs
  assert all(p + sz <= bar_size for p, sz in paddrs), f"{buf.device}: vram past the {bar_size:#x} bar"
  return [bar + p + o for p, sz in paddrs for o in range(0, sz, PAGE)]

def cut_blocks(n:int):
  """Transformer.from_gguf's models cut to their first n blocks"""
  def cut(path, parse=llm_model.gguf_parse):
    kv, entries = parse(path)
    arch = kv["general.architecture"]
    kv[f"{arch}.block_count"] = n + kv.get(f"{arch}.nextn_predict_layers", 0)
    return kv, {k: v for k, v in entries.items() if not k.startswith("blk.") or int(k.split(".")[1]) < n}
  llm_model.gguf_parse = cut

class Gpu:
  """one gpu's part of the experts: its units of every expert, a pool of SLOTS of them and the host buffers of each layer"""
  def __init__(self, dev:str, disk, units:int, part:int, n_layers:int, n_router:int, rows:int, tokens:int=0, n_experts:int=0):
    self.dev, self.disk, self.units, self.part = dev, disk, units, part
    self.pool = Tensor.empty(SLOTS * part // 2, dtype=dtypes.uint16, device=dev).realize()
    self.prps = PRPList(disk.drives[0], vram_pages(cast(Buffer, self.pool.uop.buffer)))
    self.pages, self.lists = np.array(self.prps.pages, np.int64), np.array(self.prps.addrs, np.int64)
    # the router's signal, the flags of the experts' first part and of all of them
    self.tag, self.ready1, self.ready = ([host_tensor(dev, (1,), dtypes.uint32) for _ in range(n_layers)] for _ in range(3))
    # (a prompt's chunk) every layer's experts' slots, the host's slot_of
    self.tab = host_tensor(dev, (n_layers * n_experts,), dtypes.int32)
    # for a step of T tokens (1, VERIFY's): every token's experts (slot and weight, entry i*T + t the i-th of token t) in two parts, K1 a
    # token in the first, and every token's router logits on the layer's input (of the layer, of the next: the guesses)
    self.slots:dict[int, list] = {}
    self.w:dict[int, list] = {}
    self.slots2:dict[int, list] = {}
    self.w2:dict[int, list] = {}
    self.scores:dict[int, list] = {}
    for T in sorted({1, tokens or 1}):
      self.slots[T], self.slots2[T] = ([host_tensor(dev, (n,), dtypes.int32) for _ in range(n_layers)] for n in (K1 * T, (KMAX - K1) * T))
      self.w[T], self.w2[T] = ([host_tensor(dev, (n,), dtypes.float32) for _ in range(n_layers)] for n in (K1 * T, (KMAX - K1) * T))
      self.scores[T] = [host_tensor(dev, (T * rows, n_router), dtypes.float32) for _ in range(n_layers)]

# *** the trunk as it is in vram after a load, on the store's disks: a launch reads it back (p2p, seconds) instead of unpacking the gguf

TRUNK = 1 << 40  # disk offset of the snapshot on each disk: on disk 0 its index, each gpu's buffers on its disk after EXPERTS (past the experts)

def _part(obj, name:str):
  *path, last = name.split(".")
  for p in path: obj = obj[int(p)] if p.isdigit() else obj[p] if isinstance(obj, dict) else getattr(obj, p)
  return obj, last

def _bufs(t:Tensor) -> list[Buffer]:
  b = t.uop.buffer
  return b.bufs if isinstance(b, MultiBuffer) else [b]

def _io(disk, opcode:int, off:int, buf:Buffer):
  pages, size = vram_pages(buf), round_up(buf.nbytes, PAGE)
  assert len(pages) * PAGE >= size, f"{buf.device}: a buffer of {buf.nbytes} bytes in {len(pages)} pages"
  host_io(disk, opcode, off, PRPList(disk.drives[0], pages), size)

def save_trunk(model:Transformer, names:list[str], disks:list, devs:tuple[str, ...], gguf:str):
  """every parameter (realized) to its gpus' disks, each gpu's buffers on its own, and the index"""
  sd, off = nn.state.get_state_dict(model), [TRUNK + EXPERTS] * len(disks)
  index:dict = {"gguf": os.path.basename(gguf), "size": os.path.getsize(gguf), "tensors": {}, "linears": {}}
  for name in names:
    t, parts = sd[name], []
    for b in _bufs(t):
      g = devs.index(Device.canonicalize(b.device))
      _io(disks[g], OP_WRITE, off[g], b)
      parts.append((g, off[g], b.nbytes))
      off[g] += round_up(b.nbytes, PAGE)
    index["tensors"][name] = {"shape": list(t.shape), "dtype": t.dtype.name, "axis": t.uop.axis, "parts": parts}
  # the linears' quantization, found once from the gguf's expressions (Linear.set_quantized)
  for name in {n.rsplit(".", 1)[0] for n in names}:
    if isinstance(m:=_part(model, name + ".x")[0], Linear) and m.ggml_type is not None: index["linears"][name] = [m.ggml_type, m.shard_axis]
  write_manifest(disks[0], index, TRUNK)
  print(f"saved the trunk: {len(names)} tensors, {sum(o - TRUNK - EXPERTS for o in off) / 1e9:.1f} GB", flush=True)

def trunk_index(disks:list, gguf:str) -> dict|None:
  """the snapshot's index, if there's one of this gguf"""
  try: index = read_manifest(disks[0], TRUNK)
  except ValueError: return None
  return index if index.get("gguf") == os.path.basename(gguf) and index.get("size") == os.path.getsize(gguf) else None

def load_trunk(model:Transformer, disks:list, devs:tuple[str, ...], index:dict):
  """the parameters from the snapshot: allocated at once, read from each gpu's disk into its memory"""
  ts = {name: Tensor.empty(*e["shape"], dtype=getattr(dtypes, e["dtype"]), device=devs, axis=e["axis"]) for name, e in index["tensors"].items()}
  Tensor.realize(*ts.values())
  for name, t in ts.items():
    for (g, off, _), b in zip(index["tensors"][name]["parts"], _bufs(t)): _io(disks[g], OP_READ, off, b)
    obj, last = _part(model, name)
    if isinstance(obj, dict): obj[last] = t
    else: setattr(obj, last, t)
  for name, (ggml_type, shard_axis) in index["linears"].items():
    m = _part(model, name + ".x")[0]
    m.ggml_type, m.shard_axis = ggml_type, shard_axis

# the host's part of a step in c (step_io): its state, the drives' command tables and queues, the counts in one struct, io_t (IO: name, c
# type, a pointer is an address), the gpus' buffers of a layer (BUFS) in a table of every layer's
IO = [*((n, "int64_t") for n in "layers E ngpu k kmin kmax rows tokens k1 dry nd entries slots victims_n nvict pf fill".split()),
      ("tau", "float"), ("taum", "float"), ("tauc", "float"), ("raw", "int64_t"), ("beta", "float"), ("decay", "float"), ("bufs", "const uint64_t *"),
      ("bias", "const float **"), ("tabs", "const uint64_t *"),
      ("slot_of", "int32_t *"),
      ("expert_in", "int64_t *"),
      ("freq", "float *"), ("flying", "int32_t *"), ("guess", "uint8_t *"), ("mask", "uint8_t *"), ("victims", "int64_t *"),
      ("owner", "int64_t *"), ("ptr", "const int64_t *"), ("drive", "const int32_t *"), ("lba", "const int64_t *"), ("nlb", "const uint32_t *"),
      ("kind", "const int8_t *"), *((n, "const int64_t *") for n in "inpart part_pages page_base list_base pages lists".split()),
      *((n, "const uint64_t *") for n in "sq db cq cq_db".split()), *((n, "int64_t *") for n in "tails db_tails heads phases inflight".split()),
      *((n, "int64_t") for n in "refs hits late misses guesses used wasted fills".split()),
      *((n, "double") for n in "t_gpu t_route t_reads t_after slowest".split())]
BUFS = ["tag", "ready", "ready1", "scores", "slots", "ws", "slots2", "ws2"]
CTYPES:dict[str, type[ctypes._SimpleCData]] = {"int64_t": ctypes.c_int64, "float": ctypes.c_float, "double": ctypes.c_double}
class IoT(ctypes.Structure): _fields_ = [(n, ctypes.c_void_p if "*" in t else CTYPES[t]) for n, t in IO]
HOST_C = "#include <stdint.h>\ntypedef struct { " + " ".join(f"{t} {n};" for n, t in IO) + " } io_t;\nenum { " + \
  ", ".join(f"B_{b.upper()}" for b in BUFS) + ", NB };\n" + r"""
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
typedef struct { uint32_t cdw0, nsid; uint64_t rsvd, mptr, prp1, prp2; uint32_t cdw10, cdw11, cdw12, pad[3]; } sqe_t;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

// the reads of experts codes (layer * E + expert) into slots on queue (0: demand, 1: guesses) of their drives, rung later. a command's
// owner is its code when owned (guesses, a chunk's reads), else -1: flying counts an owned one's commands till they're reaped
static void submit(io_t *s, int64_t n, const int64_t *codes, const int64_t *slots, int64_t queue, int64_t owned) {
  for (int64_t i = 0; i < n; i++) for (int64_t r = s->ptr[codes[i]]; r < s->ptr[codes[i] + 1]; r++) {
    int64_t page = slots[i] * s->part_pages[r] + s->inpart[r], gp = s->page_base[r] + page, q = s->drive[r] + queue * s->nd, cid = s->tails[q];
    sqe_t *e = (sqe_t *)s->sq[q] + cid;
    e->cdw0 = 2 | (uint32_t)cid << 16; e->nsid = 1; e->rsvd = 0; e->mptr = 0; e->prp1 = s->pages[gp];
    e->prp2 = s->kind[r] == 0 ? 0 : s->kind[r] == 1 ? s->pages[gp + 1] : s->lists[s->list_base[r] + (page + 1) / 256] + (page + 1) % 256 * 8;
    e->cdw10 = (uint32_t)s->lba[r]; e->cdw11 = (uint32_t)(s->lba[r] >> 32); e->cdw12 = s->nlb[r];
    s->owner[q * s->entries + cid] = owned ? codes[i] : -1;
    s->tails[q] = (cid + 1) % s->entries, s->inflight[q]++;
    if (owned) s->flying[codes[i]]++;
  }
}

// the doorbells of the demand queues and of the guess queues of drives without demand reads: a drive reads the guesses once its
// demand is in. only the queues that moved (a doorbell write that doesn't move a queue wedges the controller)
static void ring(io_t *s) {
  __sync_synchronize();  // the entries before the doorbells
  for (int64_t q = 0; q < 2 * s->nd; q++) if (s->tails[q] != s->db_tails[q] && (q < s->nd || !s->inflight[q - s->nd]))
    *(volatile uint32_t *)s->db[q] = (uint32_t)(s->db_tails[q] = s->tails[q]);
}

// the new completions of every queue: heads and phases on, the completion doorbells written, inflight and the guesses' flying down. -1
// when a command failed
static int64_t reap(io_t *s) {
  for (int64_t q = 0; q < 2 * s->nd; q++) {
    int64_t h = s->heads[q], ph = s->phases[q], m = 0;
    const volatile uint32_t *c = (const volatile uint32_t *)s->cq[q];
    uint32_t dw3;
    while (((dw3 = c[h * 4 + 3]) >> 16 & 1) == ph) {
      if (dw3 >> 17) return -1;
      int64_t o = s->owner[q * s->entries + (dw3 & 0xffff)];
      if (o >= 0) s->flying[o]--;
      m++;
      if (++h == s->entries) h = 0, ph ^= 1;
    }
    if (m) s->heads[q] = h, s->phases[q] = ph, *(volatile uint32_t *)s->cq_db[q] = (uint32_t)h, s->inflight[q] -= m;
  }
  return 0;
}

// the k experts of the largest sigmoid + bias (+ beta those with a slot), weights normalized, heaviest first; keep-resident truncation
// keeps the top kmin, those of weight >= tau and (resident) those with a slot, but those under tauc, at most kmax, renormalized (raw:
// not; taum: those left out lightest first, while their weight is at most taum); the rest without a slot go on skip (if any, ns of them).
// sig: every expert's sigmoid (one loop: vectorized, libmvec). returns the number kept
static int64_t route(const io_t *s, const float *logits, const float *bias, const int32_t *slot, int64_t resident, int64_t *sel, float *w,
                     float *sig, int64_t *skip, int64_t *ns) {
  int64_t n = s->E, k = s->k, kmin = s->kmin, kmax = s->kmax, idx[64], m = 0; float tau = s->tau, key[64], sv[64];
  for (int64_t i = 0; i < n; i++) sig[i] = 1.0f / (1.0f + __builtin_expf(-logits[i]));
  for (int64_t i = 0; i < n; i++) {
    float v = sig[i], kk = v + bias[i] + (slot[i] >= 0 ? s->beta : 0);
    if (m == k && kk <= key[m - 1]) continue;
    int64_t j = m < k ? m++ : m - 1;
    for (; j > 0 && key[j - 1] < kk; j--) key[j] = key[j - 1], sv[j] = sv[j - 1], idx[j] = idx[j - 1];
    key[j] = kk, sv[j] = v, idx[j] = i;
  }
  float sum = 0;
  for (int64_t j = 0; j < m; j++) sum += sv[j];
  for (int64_t a = 1; a < m; a++) for (int64_t j = a; j > 0 && sv[j - 1] < sv[j]; j--) {  // by weight
    float t = sv[j]; sv[j] = sv[j - 1], sv[j - 1] = t; int64_t u = idx[j]; idx[j] = idx[j - 1], idx[j - 1] = u;
  }
  int64_t c = 0; float kept = 0, lost = 0; uint8_t out[64];
  for (int64_t j = m - 1; j >= 0; j--) {
    out[j] = j >= kmin && (sv[j] / sum < s->tauc ||
                           (tau && sv[j] / sum < tau && !(resident && slot[idx[j]] >= 0) && (!s->taum || lost + sv[j] / sum <= s->taum)));
    if (out[j]) lost += sv[j] / sum;
  }
  for (int64_t j = 0; j < m; j++)
    if (c < kmax && !out[j]) sel[c] = idx[j], w[c] = sv[j], kept += sv[j], c++;
    else if (skip && slot[idx[j]] < 0) skip[(*ns)++] = idx[j];
  for (int64_t j = 0; j < c; j++) w[j] /= s->raw ? sum : kept;
  return c;
}

// a slot a victim can be: not masked (pinned), not of an expert in flight or guessed
static int64_t free_slot(const io_t *s, int64_t sl) {
  int64_t c = s->expert_in[sl];
  return !s->mask[sl] && !(c >= 0 && (s->flying[c] || s->guess[c]));
}

// the victims of the next misses and guesses: the n least used free slots (an empty one first), least used first
static void pick(io_t *s, int64_t n) {
  float key[512]; int64_t m = 0;
  for (int64_t sl = 0; sl < s->slots; sl++) {
    int64_t c = s->expert_in[sl];
    if (!free_slot(s, sl)) continue;
    float f = c >= 0 ? s->freq[c] : -1;
    if (m == n && f >= key[m - 1]) continue;
    int64_t j = m < n ? m++ : m - 1;
    for (; j > 0 && key[j - 1] > f; j--) key[j] = key[j - 1], s->victims[j] = s->victims[j - 1];
    key[j] = f, s->victims[j] = sl;
  }
  s->nvict = m;
}

// the first victim still free, taken out of the list, into code's slot (-1: none left)
static int64_t take(io_t *s, int64_t code) {
  for (int64_t i = 0; i < s->nvict; i++) if (free_slot(s, s->victims[i])) {
    int64_t sl = s->victims[i];
    for (s->nvict--; i < s->nvict; i++) s->victims[i] = s->victims[i + 1];
    if (s->expert_in[sl] >= 0) s->slot_of[s->expert_in[sl]] = -1;
    s->expert_in[sl] = code, s->slot_of[code] = (int32_t)sl;
    return sl;
  }
  return -1;
}

// row of the gpus' scores of a layer (b its buffers), E / ngpu experts a gpu
static void logits_of(io_t *s, const uint64_t *b, int64_t row, float *out) {
  for (int64_t g = 0, nr = s->E / s->ngpu; g < s->ngpu; g++)
    for (int64_t i = 0; i < nr; i++) out[g * nr + i] = ((const float *)b[B_SCORES * s->ngpu + g])[row * nr + i];
}

// the guesses of layer lj on a row of the scores of the layer of buffers b: the first n its router would keep that aren't cached, in the
// router's order (sigmoid + bias: a guess high in it is likelier routed), read into victims on the guess queues, held till lj routes
static void prefetch(io_t *s, const uint64_t *b, int64_t lj, int64_t row, int64_t n) {
  float logits[4096], sig[4096], w[64]; int64_t sel[64], codes[64], ms[64], m = 0;
  logits_of(s, b, row, logits);
  const float *bias = s->bias[lj];
  int32_t *so = s->slot_of + lj * s->E;
  int64_t c = route(s, logits, bias, so, 0, sel, w, sig, 0, 0);
  #define KEY(e) (sig[e] + bias[e])
  for (int64_t a = 1; a < c; a++) for (int64_t j = a; j > 0 && KEY(sel[j - 1]) < KEY(sel[j]); j--) {
    int64_t u = sel[j]; sel[j] = sel[j - 1], sel[j - 1] = u;
  }
  for (int64_t j = 0; j < c && m < n; j++) {
    if (so[sel[j]] >= 0) continue;
    if ((ms[m] = take(s, codes[m] = lj * s->E + sel[j])) < 0) break;
    s->guess[codes[m++]] = 1;
  }
  submit(s, m, codes, ms, 1, 1);
  s->guesses += m;
}

// layer li of a step of tokens (each its rows of the gpus' scores): waits for the gpus' router signals, routes every token on its row 0
// (the union of their experts read once), reads the misses into victims, queues the guesses of the next layer (row 1, only while every
// guess queue is under half full: they can't wrap one, the demand's are drained every layer), gives the gpus every token's slots and
// weights in two parts, each in the order they're ready (cached, cached and in flight, read now): the first K1 a token, those in (one
// in flight or read only when the second can't hold the rest), go while the rest are read; entry i*T + t the i-th of token t, padded
// with slot 0, weight 0 (the gemvs skip it).
// releases them. then the counts, li's guesses settled (used when routed), the decayed use counts and the next layer's victims (but this layer's
// slots). dry: no reads, the slots hold whatever they hold.
// 0, -1 a read failed, -2 a timeout, -3 no victims
static int64_t layer(io_t *s, int64_t li) {
  int64_t E = s->E, ng = s->ngpu, T = s->tokens, nd = s->nd, K1 = s->k1 / T, K2 = s->kmax - K1;
  const uint64_t *b = s->bufs + li * NB * ng;
  #define BUF(kind, g, type) ((type *)b[(kind) * ng + (g)])
  double t0 = now();
  for (int64_t g = 0; g < ng; g++) while (*BUF(B_TAG, g, volatile uint32_t) != 1) {  // (the guesses go on)
    if (reap(s) < 0) return -1;
    ring(s);
    if (now() - t0 > 60) return -2;
  }
  double t1 = now();
  float logits[4096], sig[4096], tw[512];
  int64_t sel[64], pos[4096], usel[512], codes[512], ms[512], skip[512], tj[512], tc[8], p1[8], nu = 0, nm = 0, nh = 0, late = 0, ns = 0,
          nf = 0;
  uint8_t hit[512];
  int32_t *so = s->slot_of + li * E;
  for (int64_t e = 0; e < E; e++) pos[e] = -1;
  for (int64_t t = 0; t < T; t++) {
    logits_of(s, b, t * s->rows, logits);
    int64_t c = tc[t] = route(s, logits, s->bias[li], so, 1, sel, tw + t * 64, sig, skip, &ns);
    for (int64_t j = 0; j < c; j++) {
      if (pos[sel[j]] < 0) pos[sel[j]] = nu, usel[nu++] = sel[j];
      tj[t * 64 + j] = pos[sel[j]];
    }
  }
  for (int64_t j = 0; j < nu; j++) {
    if ((hit[j] = s->dry || so[usel[j]] >= 0)) nh++, late += s->flying[li * E + usel[j]] > 0;
    else codes[nm++] = li * E + usel[j];
    if (!s->dry && hit[j]) s->mask[so[usel[j]]] = 1;
  }
  if (!s->dry) {
    int64_t free = 0;
    for (int64_t i = 0; i < s->nvict; i++) free += free_slot(s, s->victims[i]);
    if (free < nm) pick(s, s->victims_n);  // too few: new ones, but this layer's cached
    for (int64_t i = 0; i < nm; i++) if ((ms[i] = take(s, codes[i])) < 0) return -3;
    submit(s, nm, codes, ms, 0, 0);
    ring(s);
  }
  double t2 = now();
  #define FLYING(u) (hit[u] && s->flying[li * E + usel[u]] > 0)
  #define RANK(u) (!hit[u] ? 2 : FLYING(u) ? 1 : 0)
  #define BUSY(u) (hit[u] ? FLYING(u) : reads)
  // each token's experts by readiness (stable), its part 1 those in
  for (int64_t t = 0; t < T; t++) {
    int64_t *o = tj + t * 64, c = tc[t], nht = 0; float *ow = tw + t * 64;
    for (int64_t a = 1; a < c; a++) for (int64_t j = a; j > 0 && RANK(o[j - 1]) > RANK(o[j]); j--) {
      int64_t u = o[j]; o[j] = o[j - 1], o[j - 1] = u; float x = ow[j]; ow[j] = ow[j - 1], ow[j - 1] = x;
    }
    for (int64_t j = 0; j < c; j++) nht += RANK(o[j]) == 0;
    p1[t] = nht < K1 ? nht : K1;
    if (c - p1[t] > K2) p1[t] = c - K2;
  }
  for (int64_t g = 0; g < ng; g++) for (int64_t i = 0; i < s->kmax * T; i++) {
    int64_t second = i >= K1 * T, at = second ? i - K1 * T : i, t = at % T, n = at / T + (second ? p1[t] : 0);
    int64_t on = n < (second ? tc[t] : p1[t]);
    BUF(second ? B_SLOTS2 : B_SLOTS, g, int32_t)[at] = (int32_t)(!on ? 0 : s->dry ? i % s->slots : so[usel[tj[t * 64 + n]]]);
    BUF(second ? B_WS2 : B_WS, g, float)[at] = on ? tw[t * 64 + n] : 0;
  }
  for (int64_t r1 = 0, first = 1;; first = 0) {
    if (reap(s) < 0) return -1;
    ring(s);
    int64_t reads = 0, busy = 0, busy1 = 0;
    for (int64_t q = 0; q < nd; q++) reads |= s->inflight[q] != 0;
    for (int64_t u = 0; u < nu; u++) busy |= BUSY(u);
    for (int64_t t = 0; t < T; t++) for (int64_t n = 0; n < p1[t]; n++) busy1 |= BUSY(tj[t * 64 + n]);
    if (!r1 && !busy1) {
      __sync_synchronize();  // the slots and weights before the gpus go on
      for (int64_t g = 0; g < ng; g++) *BUF(B_READY1, g, volatile uint32_t) = 1;
      r1 = 1;
    }
    if (first && !s->dry) {  // (after part 1 is out: the gpus work on it) the next layer's guesses, the fills, this layer's hits unpinned
      int64_t room = 1;
      for (int64_t q = nd; q < 2 * nd; q++) room &= s->inflight[q] < s->entries / 2;
      if (room && li + 1 < s->layers) for (int64_t t = 0; t < T; t++) prefetch(s, b, li + 1, t * s->rows + 1, s->pf);
      for (int64_t i = 0; room && i < ns && nf < s->fill; i++) if (pos[skip[i]] == -1) {  // (each once, none another token kept)
        pos[skip[i]] = -2;
        if ((ms[nm + nf] = take(s, codes[nm + nf] = li * E + skip[i])) < 0) break;
        nf++;
      }
      submit(s, nf, codes + nm, ms + nm, 1, 1);
      ring(s);
      for (int64_t j = 0; j < nu; j++) if (hit[j]) s->mask[so[usel[j]]] = 0;
    }
    if (!busy) break;
    if (now() - t0 > 60) return -2;
  }
  __sync_synchronize();
  for (int64_t g = 0; g < ng; g++) *BUF(B_READY1, g, volatile uint32_t) = 1, *BUF(B_READY, g, volatile uint32_t) = 1;
  double t3 = now();
  s->refs += nu, s->hits += nh - late, s->late += late, s->misses += nm, s->fills += nf;
  for (int64_t e = 0; e < E; e++) if (s->guess[li * E + e]) {
    s->guess[li * E + e] = 0;
    if (pos[e] >= 0) s->used++; else s->wasted++;
  }
  float *f = s->freq + li * E;
  for (int64_t e = 0; e < E; e++) f[e] *= s->decay;
  for (int64_t j = 0; j < nu; j++) f[usel[j]] += 1;
  for (int64_t i = 0; i < nf; i++) f[codes[nm + i] - li * E] += 1;  // (a fill counts as routed: it was)
  if (!s->dry) {
    for (int64_t j = 0; j < nu; j++) s->mask[so[usel[j]]] = 1;
    pick(s, s->victims_n);
    for (int64_t j = 0; j < nu; j++) s->mask[so[usel[j]]] = 0;
  }
  s->t_gpu += t1 - t0, s->t_route += t2 - t1, s->t_reads += t3 - t2, s->t_after += now() - t3;
  return 0;
}

// every layer's ready flags (the first parts' too) set to value, with 0 the router tags cleared too
void flags(io_t *s, uint32_t value) {
  for (int64_t li = 0; li < s->layers; li++) for (int64_t g = 0, ng = s->ngpu; g < ng; g++) {
    const uint64_t *b = s->bufs + li * NB * ng;
    *(volatile uint32_t *)b[B_READY * ng + g] = value, *(volatile uint32_t *)b[B_READY1 * ng + g] = value;
    if (!value) *(volatile uint32_t *)b[B_TAG * ng + g] = 0;
  }
}

// every layer of a step: the host's whole part of it while the gpus run the graph. 0, else layer's error. slowest: its slowest layer
int64_t step_io(io_t *s) {
  s->slowest = 0;
  for (int64_t li = 0; li < s->layers; li++) {
    double t = now();
    int64_t r = layer(s, li);
    if (r) return r;
    if (now() - t > s->slowest) s->slowest = now() - t;
  }
  return 0;
}

// *** a chunk of a prompt (many tokens a step): the gpus route it themselves, the host reads every expert of a layer into a slot and gives
// the gpus the layer's slots (its row of tabs). layer l+1's reads go while the gpus work on layer l (from when they're at l's router: its
// tag, l-1's slots free then), the reads not queued yet wait in pend (a drive's queues hold 1024)

static int64_t pend[8192], npend, pend_at;

// the pending reads queued while a queue of every drive of the next is under full (the demand queue, then the guess one), each command
// owned by its code: flying counts them till they're in
static void drain(io_t *s) {
  for (; pend_at < npend; pend_at++) {
    int64_t c = pend[pend_at], sl = s->slot_of[c], q = -1;
    for (int64_t k = 0; k < 2 && q < 0; k++) {
      int64_t ok = 1;
      for (int64_t r = s->ptr[c]; r < s->ptr[c + 1]; r++) ok &= s->inflight[s->drive[r] + k * s->nd] < s->entries - 8;
      if (ok) q = k;
    }
    if (q < 0) return;
    submit(s, 1, &c, &sl, q, 1);
  }
}

static void ring_all(io_t *s) {
  __sync_synchronize();
  for (int64_t q = 0; q < 2 * s->nd; q++)
    if (s->tails[q] != s->db_tails[q]) *(volatile uint32_t *)s->db[q] = (uint32_t)(s->db_tails[q] = s->tails[q]);
}

typedef struct { float f; int32_t sl; } slotkey_t;
static int cmp_key(const void *a, const void *b) { float x = ((const slotkey_t *)a)->f, y = ((const slotkey_t *)b)->f; return x < y ? -1 : x > y; }

// layer lj's experts without a slot pending, into the least used slots not of layer lj or li (the gpus' now, -1: none). -3: too few slots
static int64_t queue_layer(io_t *s, int64_t lj, int64_t li) {
  int64_t E = s->E, n = 0;
  static slotkey_t keys[16384];
  for (int64_t e = 0; e < E; e++) {
    if (s->slot_of[lj * E + e] >= 0) s->mask[s->slot_of[lj * E + e]] = 1;
    if (li >= 0 && s->slot_of[li * E + e] >= 0) s->mask[s->slot_of[li * E + e]] = 1;
  }
  for (int64_t sl = 0; sl < s->slots; sl++) if (free_slot(s, sl)) {
    int64_t c = s->expert_in[sl];
    keys[n].f = c >= 0 ? s->freq[c] : -1, keys[n++].sl = (int32_t)sl;
  }
  qsort(keys, n, sizeof(slotkey_t), cmp_key);
  int64_t v = 0, ret = 0;
  for (int64_t e = 0; e < E; e++) {
    int64_t c = lj * E + e;
    if (s->slot_of[c] >= 0) continue;
    if (v == n) { ret = -3; break; }
    int64_t sl = keys[v++].sl;
    if (s->expert_in[sl] >= 0) s->slot_of[s->expert_in[sl]] = -1;
    s->expert_in[sl] = c, s->slot_of[c] = (int32_t)sl;
    pend[npend++] = c;
    s->misses++;
  }
  s->refs += E;
  for (int64_t sl = 0; sl < s->slots; sl++) s->mask[sl] = 0;
  return ret;
}

// every layer of a chunk's step. 0, -1 a read failed, -2 a timeout, -3 no slots
int64_t chunk_io(io_t *s) {
  int64_t E = s->E, ng = s->ngpu, r;
  npend = pend_at = 0;
  if ((r = queue_layer(s, 0, -1))) return r;
  for (int64_t li = 0; li < s->layers; li++) {
    const uint64_t *b = s->bufs + li * NB * ng;
    double t0 = now();
    for (int64_t g = 0; g < ng; g++) while (*BUF(B_TAG, g, volatile uint32_t) != 1) {
      if (reap(s) < 0) return -1;
      drain(s), ring_all(s);
      if (now() - t0 > 60) return -2;
    }
    double t1 = now();
    if (li + 1 < s->layers && (r = queue_layer(s, li + 1, li))) return r;
    double t2 = now();
    for (int64_t busy = 1; busy;) {
      if (reap(s) < 0) return -1;
      drain(s), ring_all(s);
      busy = 0;
      for (int64_t e = 0; e < E && !busy; e++) busy = s->flying[li * E + e] > 0;
      if (now() - t0 > 60) return -2;
    }
    for (int64_t g = 0; g < ng; g++) for (int64_t e = 0; e < E; e++) ((int32_t *)s->tabs[g])[li * E + e] = s->slot_of[li * E + e];
    __sync_synchronize();
    for (int64_t g = 0; g < ng; g++) *BUF(B_READY, g, volatile uint32_t) = 1;
    s->t_gpu += t1 - t0, s->t_route += t2 - t1, s->t_reads += now() - t2;
  }
  return 0;
}
"""

@functools.cache
def host_c():
  import hashlib, subprocess, tempfile
  so = os.path.join(tempfile.gettempdir(), f"k3_host_{hashlib.md5(HOST_C.encode()).hexdigest()}.so")
  if not os.path.exists(so):  # (the sigmoids in vector expfs: libmvec)
    subprocess.run(["clang", "-O2", "-march=native", "-fno-math-errno", "-fveclib=libmvec", "-shared", "-fPIC", "-x", "c", "-", "-o", so, "-lmvec"],
                   input=HOST_C.encode(), check=True)
  lib = ctypes.CDLL(so)
  lib.step_io.restype, lib.step_io.argtypes = ctypes.c_int64, [ctypes.POINTER(IoT)]
  lib.chunk_io.restype, lib.chunk_io.argtypes = ctypes.c_int64, [ctypes.POINTER(IoT)]
  lib.flags.argtypes = [ctypes.POINTER(IoT), ctypes.c_uint32]
  return lib

class Reads:
  """the nvme reads of expert parts into the gpu pools. the split of a part into drive commands is static: a table of the commands of
  every expert (layer * E + expert) on all gpus is built once, step_io queues them. queue j of drive k is queues[j * len(drives) + k]"""
  def __init__(self, gpus:list[Gpu], n_codes:int, experts_off:int):
    self.drives = [d for gpu in gpus for d in gpu.disk.drives]
    self.entries, shift = self.drives[0].io_q.entries, self.drives[0].lba_shift
    queues = [d.io_qs[j] for j in range(len(self.drives[0].io_qs)) for d in self.drives]
    rows, codes = [], np.arange(n_codes, dtype=np.int64)
    for g, gpu in enumerate(gpus):
      start, nd = experts_off + codes * gpu.part, len(gpu.disk.drives)
      s0, cnt = start // STRIPE, (start + gpu.part - 1) // STRIPE - start // STRIPE + 1
      x = np.repeat(codes, cnt)
      stripe = s0[x] + np.arange(len(x)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
      lo, hi = np.maximum(stripe * STRIPE, start[x]), np.minimum((stripe + 1) * STRIPE, start[x] + gpu.part)
      rows.append((x, g * nd + stripe % nd, ((stripe // nd) * STRIPE + lo % STRIPE) >> shift, hi - lo, (lo - start[x]) // PAGE, np.full(len(x), g)))
    cols = [np.concatenate(c) for c in zip(*rows)]
    code, drive, lba, nbytes, inpart, gid = (c[np.argsort(cols[0], kind="stable")] for c in cols)
    def i64(a): return np.ascontiguousarray(a, np.int64)
    def addr(mv): return ctypes.addressof(ctypes.c_char.from_buffer(mv))
    # per command: its drive, lba, blocks - 1, prp kind (1 page: prp2 0, 2: the second page, more: a list), page in the part, its gpu's
    # pool pages per part, first page and first list
    self.tables = {"ptr": i64(np.searchsorted(code, np.arange(n_codes + 1))), "drive": drive.astype(np.int32), "lba": i64(lba),
                   "nlb": ((nbytes >> shift) - 1).astype(np.uint32), "kind": np.minimum((nbytes + PAGE - 1) // PAGE - 1, 2).astype(np.int8),
                   "inpart": i64(inpart), "part_pages": i64(np.array([gpu.part // PAGE for gpu in gpus])[gid]),
                   "page_base": i64(np.cumsum([0] + [len(gpu.pages) for gpu in gpus])[gid]),
                   "list_base": i64(np.cumsum([0] + [len(gpu.lists) for gpu in gpus])[gid]),
                   "pages": i64(np.concatenate([gpu.pages for gpu in gpus] + [[0]])),
                   "lists": i64(np.concatenate([gpu.lists for gpu in gpus] + [[0]])),
                   # every queue's ring, doorbells and state (only step_io pushes on them from here on), each command's guess (-1: demand)
                   "sq": np.array([addr(q.sq) for q in queues], np.uint64),
                   "db": np.array([q.dev.regs.addr + q.sq_db * 4 for q in queues], np.uint64),
                   "cq": np.array([addr(q.cq.cast("B")) for q in queues], np.uint64),
                   "cq_db": np.array([q.dev.regs.addr + q.cq_db * 4 for q in queues], np.uint64),
                   **{k: i64([getattr(q, a) for q in queues]) for k, a in (("tails", "tail"), ("db_tails", "db_tail"), ("heads", "head"),
                                                                           ("phases", "phase"), ("inflight", "inflight"))},
                   "owner": np.full(len(queues) * self.entries, -1, np.int64)}

class Streamer:
  def __init__(self, model:Transformer, disks:list, m:dict):
    self.disks, self.m, self.model, self.vocab = disks, m, model, model.config.vocab_size
    self.E, self.lat, units, parts = m["n_experts"], m["lat"], m["units"], m["part_bytes"]
    blk, tsz = m["block"]
    self.row_up, self.qtype = self.lat // blk * tsz, {n: t for t, n in QUANT_NAMES.items()}[m.get("expert_type", "IQ3_XXS").lower()]
    self.blocks = [model.blk[L] for L in m["layers"] if L < len(model.blk)]  # BLOCKS=n: a model cut short has the first layers
    devs = self.blocks[0].ffn_gate_inp.weight.device
    assert len(devs) == len(units) and self.E % len(devs) == 0
    self.T = getenv("VERIFY", 2 if getenv("DRAFT", "") else 0)  # tokens of a verify step (0: none)
    self.gpus = [Gpu(d, disk, u, p, len(self.blocks), self.E // len(devs), ROWS, self.T, self.E)
                 for d, disk, u, p in zip(devs, self.disks, units, parts)]
    self.expert_bytes = sum(g.part for g in self.gpus)
    # a layer's router and the guess of the next layer's experts on its input (GUESS=dir: maps fitted on the traces, extra/k3/train_guess.py
    # guess-T.npy, T the block; else that layer's router) in one q8_0 gemv on each gpu, its rows of the gpu's experts
    gdir, L, n = getenv("GUESS", ""), len(self.blocks), self.E // len(devs)
    routers = [b.ffn_gate_inp.weight.float().numpy() for b in self.blocks]
    def guess(li:int) -> np.ndarray:
      return np.load(f).astype(np.float32) if li < L and os.path.exists(f:=f"{gdir}/guess-{m['layers'][li]}.npy") else routers[min(li, L - 1)]
    self.scorers = [[q8_0_linear(np.concatenate([routers[li][g * n:(g + 1) * n], guess(li + 1)[g * n:(g + 1) * n]]), devs[g])
                     for g in range(len(devs))] for li in range(L)]
    for b in self.blocks: b.ffn_gate_inp.weight = Tensor.zeros(1)  # (the streamed path routes on the scorers)
    self.bias = [(b.exp_probs_b["bias"].float().numpy().view(np.uint32) + 0x8000 & 0xffff0000).view(np.float32) for b in self.blocks]
    # the host's state, used and kept by step_io: the slot of each (layer, expert) (-1 when not cached), the layer * E + expert in each
    # slot (-1 when free), the decayed use counts, the commands in flight of each guess, the guesses their layer hasn't routed (they keep
    # their slots till then), pinned slots and the victims of the next misses and guesses
    L, LE = len(self.blocks), len(self.blocks) * self.E
    self.state = {"slot_of": np.full(LE, -1, np.int32), "expert_in": np.full(SLOTS, -1, np.int64), "freq": np.zeros(LE, np.float32),
                  "flying": np.zeros(LE, np.int32), "guess": np.zeros(LE, np.uint8), "mask": np.zeros(SLOTS, np.uint8),
                  "victims": np.zeros(VICTIMS * max(self.T, 1), np.int64), "bias": np.array([b.ctypes.data for b in self.bias], np.uint64)}
    self.reads = Reads(self.gpus, LE, m["experts_off"])
    self.io = IoT(layers=L, E=self.E, ngpu=len(self.gpus), k=K, kmin=KMIN, kmax=KMAX, rows=ROWS, slots=SLOTS)
    for name, a in {**self.state, **self.reads.tables}.items(): setattr(self.io, name, a.ctypes.data)
    self.io.nd, self.io.entries = len(self.reads.drives), self.reads.entries
    # the gpus' buffers (BUFS) of every layer, (layers, BUFS, gpus), for a step of a token and (VERIFY) of T
    def table(bufs) -> np.ndarray:
      ptrs = [[[b[1].ctypes.data for b in bufs(g, li)] for g in self.gpus] for li in range(L)]
      return np.ascontiguousarray(np.array(ptrs, np.uint64).transpose(0, 2, 1))
    self.bufs = {T: table(lambda g, li: [g.tag[li], g.ready[li], g.ready1[li], g.scores[T][li], g.slots[T][li], g.w[T][li], g.slots2[T][li],
                                         g.w2[T][li]]) for T in self.gpus[0].scores}
    self.tabs = np.array([g.tab[1].ctypes.data for g in self.gpus], np.uint64)
    self.io.tabs = self.tabs.ctypes.data
    # (a prompt's chunk: the gpus route it, exact) every layer's key bias on every gpu
    self.bias_t = [[Tensor(b).to(d).realize() for d in devs] for b in self.bias]
    self.io.raw, self.io.beta, self.io.tauc = not RENORM, BETA, TAUC
    self.io.tau, self.io.taum, self.io.pf, self.io.fill, self.io.decay, self.io.dry = TAU, TAUM, PREFETCH, FILL, 0.5 ** (1 / HALF_LIFE), DRY
    self.step_times, self.slowest = np.zeros(2), cast(list[float], [])
    # after: (T, the step's position) -> more tokens to read with the step's (DRAFT: the draft's guesses), launched after the layers
    self.after:Callable[[int, UOp], Tensor]|None = None  # (slowest: each step's slowest layer)
    self.tok = host_tensor(Device.DEFAULT, (1, 1), dtypes.int32)
    # (a prompt's chunks) their tokens a size, the graph of a chunk (a jit a size: the engine's main sets it)
    self.chunk_toks:dict[int, tuple[Tensor, np.ndarray]] = {}
    self.chunk_model:Callable[..., Tensor] = model
    if self.T: self.tok_t = host_tensor(Device.DEFAULT, (1, self.T), dtypes.int32)
    # the p2p allreduce mailboxes now: made during a step (the graph's first use) its synchronize waits for a gpu that waits for this host
    p2p_allreduce(devs)
    for li, b in enumerate(self.blocks): b.routed_experts = functools.partial(self.experts, li)

  # *** the graph of a layer

  def experts(self, li:int, x:Tensor, h:Tensor) -> Tensor:
    """the experts of a step of T tokens in two parts, every token its own"""
    T, outs, x = int(x.shape[1]), [], x.half()  # (an mselect of the mix kernel's output has no shape, of its half it does)
    if T > CHUNK_MIN: return self.chunk_experts(li, x, h)
    for g, gpu in enumerate(self.gpus):
      xg, hg = Tensor(x.uop.mselect(g)).reshape(T, -1), Tensor(h.uop.mselect(g)).reshape(T, self.lat).float()
      # a fresh tensor of the buffer: an assign to the kept one would chain every token's graph onto it
      scores = Tensor(gpu.scores[T][li][0].uop).assign(self.scorers[li][g](xg).reshape(T * ROWS, -1))
      signal = UOp.custom_function("hostsignal").call(gpu.tag[li][0].uop, scores.uop)
      # the first part when its experts are in, the second after it when all are (its down gemv adds the first's sum)
      out = self.part(li, gpu, hg, gpu.slots[T][li][0], gpu.w[T][li][0], gpu.ready1[li][0].uop.after(signal))
      out = self.part(li, gpu, hg, gpu.slots2[T][li][0], gpu.w2[T][li][0], gpu.ready[li][0].uop.after(out.uop), out)
      outs.append(out.reshape(1, T * self.lat))
    # every gpu has the partial sums of its units: the sum over the gpus is the allreduce
    return partial_sum(Tensor(UOp.mstack(*[o.uop for o in outs]).unshard(0))).reshape(1, T, self.lat)

  def chunk_experts(self, li:int, x:Tensor, h:Tensor) -> Tensor:
    """the experts of a prompt's chunk: every gpu routes all of its tokens (exact) on the router logits of all the gpus, the host reads in
    every expert of the layer and gives their slots (tab, once ready)"""
    T, n, outs = int(x.shape[1]), self.E // len(self.gpus), []
    logits = [self.scorers[li][g](Tensor(x.uop.mselect(g)).reshape(T, -1))[:, :n] for g in range(len(self.gpus))]
    for g, gpu in enumerate(self.gpus):
      full = Tensor.cat(*[lg.to(gpu.dev) for lg in logits], dim=1).float().contiguous()
      signal = UOp.custom_function("hostsignal").call(gpu.tag[li][0].uop, full.uop)
      sig = full.sigmoid()
      sel = (sig + self.bias_t[li][g]).topk(K)[1]
      w = sig.gather(-1, sel)
      w = w / w.sum(-1, keepdim=True)
      wait = UOp.custom_function("hostwait").call(gpu.ready[li][0].uop.after(signal), gpu.tab[0].uop)
      tab = Tensor(gpu.tab[0].uop.after(wait))[li * self.E:(li + 1) * self.E]
      # an entry an expert of a token, i*T + t the i-th of token t
      slots, w = tab[sel].T.flatten().contiguous(), w.T.flatten().contiguous()
      hg, u = Tensor(h.uop.mselect(g)).reshape(T, self.lat).float(), gpu.units
      cfg, stride = self.blocks[li].config, gpu.part // 2
      gu = expert_linear(gpu.pool, slots, hg, 2 * u, self.lat, self.qtype, stride, w=w, summed=False)
      act = q8_act(functools.partial(gated_activation, limit=cfg.swiglu_limit, situ=cfg.situ), gu, parts=2)
      y = expert_linear(gpu.pool, slots, act, self.lat, u, self.qtype, stride, 2 * u * self.row_up // 2, w=w, summed=False)
      outs.append((y.reshape(K, T, self.lat) * w.reshape(K, T, 1)).sum(0).reshape(1, T * self.lat))
    return partial_sum(Tensor(UOp.mstack(*[o.uop for o in outs]).unshard(0))).reshape(1, T, self.lat)

  def part(self, li:int, gpu:Gpu, hg:Tensor, slots:Tensor, w:Tensor, ready:UOp, bias:Tensor|None=None) -> Tensor:
    """the sum of every token's experts (entry i*T + t the i-th of token t: slot and weight), (plus bias) once the host sets ready"""
    T = int(hg.shape[0])
    wait = UOp.custom_function("hostwait").call(ready, slots.uop, w.uop)
    slots, w = Tensor(slots.uop.after(wait)), Tensor(w.uop.after(wait))
    u, stride, cfg, tokens = gpu.units, gpu.part // 2, self.blocks[li].config, T if T > 1 else 0
    gu = expert_linear(gpu.pool, slots, hg, 2 * u, self.lat, self.qtype, stride, w=w, summed=False)  # (w: the padding not computed)
    # (the act in the q8 kernel of the down gemv's input)
    act = q8_act(functools.partial(gated_activation, limit=cfg.swiglu_limit, situ=cfg.situ), gu, parts=2)
    return expert_linear(gpu.pool, slots, act, self.lat, u, self.qtype, stride, 2 * u * self.row_up // 2, tokens=tokens, w=w,
                         bias=bias).reshape(T, self.lat)

  # *** the host side of a step

  def counters(self) -> np.ndarray:
    io = self.io
    return np.array([time.perf_counter(), (io.misses + io.guesses + io.fills) * self.expert_bytes, io.refs, io.hits, io.late, io.guesses, io.used,
                     io.wasted, io.fills,
                     io.t_gpu, io.t_route, io.t_reads, io.t_after, *self.step_times])

  def report(self, name:str, c0:np.ndarray, n:int):
    """a token's time, reads, experts ready and guesses, and its time split, over the n steps since the counters c0"""
    dt, gb, refs, hits, late, pf, used, wasted, fills, *ms = (self.counters() - c0) / n
    print(f"{name}: {dt * 1e3:.1f} ms/token = {1 / dt:.1f} tok/s, {gb / 1e9:.2f} GB/token read, {hits / max(refs, 1):.1%} ready at the router, "
          f"{late / max(refs, 1):.1%} in flight, guesses per token {pf:.0f}: {used:.0f} used, {wasted:.0f} not" +
          (f", fills {fills:.0f}" if FILL else ""))
    gpu, route, io, rest, launch, tail = (t * 1e3 for t in ms)
    print(f"  per token: {gpu:.1f} ms waiting on the gpu, {route:.1f} routing and queueing reads, {io:.1f} waiting on reads, "
          f"{rest:.1f} bookkeeping under the gpu, {launch:.1f} launching the graph, {tail:.1f} after the last layer", flush=True)
    worst = np.array(self.slowest[-n:]) * 1e3
    print(f"  a step's slowest layer: median {np.median(worst):.2f} ms, 90% {np.percentile(worst, 90):.2f} ms, max {worst.max():.1f} ms", flush=True)

  def flags(self, value:int): host_c().flags(ctypes.byref(self.io), value)

  def step(self, tok:int, pos:int, v_pos:UOp, temp:Tensor) -> int: return self.steps([tok], pos, v_pos, temp)[0]

  def chunk(self, toks:list[int], pos:int, v_pos:UOp, temp:Tensor):
    """a step of a prompt's chunk at positions pos..: the gpus run the graph (chunk_model, a graph a size), the host reads in every layer's
    experts (chunk_io)"""
    io, T = self.io, len(toks)
    io.tokens, io.bufs = T, self.bufs[1].ctypes.data
    self.flags(0)
    if T not in self.chunk_toks: self.chunk_toks[T] = host_tensor(Device.DEFAULT, (1, T), dtypes.int32)
    buf, arr = self.chunk_toks[T]
    arr[0, :] = toks
    out = self.chunk_model(buf, v_pos.bind(pos), temp)
    if (err:=host_c().chunk_io(ctypes.byref(io))):
      self.flags(1)
      raise RuntimeError({-1: "a read failed", -2: "timeout", -3: "no slots left"}[err])
    more = self.after(T, v_pos.bind(pos)) if self.after is not None else None
    for o in (out, more):
      if o is not None: o.numpy()

  def steps(self, toks:list[int], pos:int, v_pos:UOp, temp:Tensor) -> list[int]:
    """a step of tokens at positions pos.. (more than one: a verify step, VERIFY=T): the gpus run the graph, the host every layer's
    routing and reads (step_io). the greedy next token after each"""
    io, T, t0 = self.io, len(toks), time.perf_counter()
    io.tokens, io.bufs, io.k1, io.victims_n = T, self.bufs[T].ctypes.data, K1 * T, VICTIMS * T
    self.flags(0)
    buf, arr = self.tok if T == 1 else self.tok_t
    arr[0, :] = toks  # a realized input: realizing a new one walks every live tensor, 25 ms
    out = self.model(buf, v_pos.bind(pos), temp)
    t1 = time.perf_counter()
    if (err:=host_c().step_io(ctypes.byref(io))):  # let the gpus run the graph out: one killed waiting on a layer needs a power cycle
      self.flags(1)
      raise RuntimeError({-1: "a read failed", -2: "timeout", -3: "no victims left"}[err])
    t2 = time.perf_counter()
    more = self.after(T, v_pos.bind(pos)) if self.after is not None else None  # (launched now: it runs behind the step's tail)
    ret = out.numpy().reshape(-1).tolist() + (more.numpy().reshape(-1).tolist() if more is not None else [])
    # (NaN logits: argmax past the vocab, e.g. a p2p allreduce past its deadline. DRY: the slots are garbage, NaN or not)
    if max(ret) >= self.vocab:
      if not DRY: raise RuntimeError(f"step at {pos}: NaN logits, token {max(ret)}")
      ret = [min(r, self.vocab - 1) for r in ret]
    self.step_times += np.array([t1 - t0, time.perf_counter() - t2])  # the graph launch, the tail after the last layer
    self.slowest.append(io.slowest)
    return ret

if __name__ == "__main__":
  from tinygrad.llm.cli import SimpleTokenizer
  if (NB:=getenv("BLOCKS", 0)): cut_blocks(NB)  # BLOCKS=n: the model cut to its first n blocks, a quick check of the engine
  st = time.perf_counter()
  devs = tuple(Device.canonicalize(f"{Device.DEFAULT}:{i}") for i in range(getenv("SHARD", 4)))
  # a queue per drive for the reads needed now, one for guesses
  disks, m = open_store(queues=2, at=STORE)
  # the trunk from its snapshot on the disks (the gguf's tensors aren't unpacked: placeholders), else unpacked from the gguf, then saved
  # (SNAP=0: neither; a cut model neither)
  snap = trunk_index(disks, sys.argv[1]) if not NB and getenv("SNAP", 1) else None
  if snap is not None:
    llm_model.gguf_shard = lambda entries, devices, shard_map=None: {name: Tensor.empty(*shape, dtype=dtypes.float16, device=devices,
      axis=(shard_map or {}).get(name)) for name, (_, shape, _) in entries.items()}
  model, kv = Transformer.from_gguf(sys.argv[1], max_context=getenv("CTX", 4096), shard=getenv("SHARD", 4))
  tok = SimpleTokenizer.from_gguf_kv(kv)
  print(f"model in {time.perf_counter() - st:.0f} s", flush=True)
  names = list(nn.state.get_state_dict(model))
  if snap is not None: load_trunk(model, disks, devs, snap)
  elif not NB and getenv("SNAP", 1):
    for name in names:
      if isinstance(l:=_part(model, name)[0], Linear) and l.ggml_type is None and l.use_custom_quant: l.set_quantized(l.weight)
    sd = nn.state.get_state_dict(model)
    Tensor.realize(*[sd[n] for n in names])
    save_trunk(model, names, disks, devs, sys.argv[1])
  print(f"trunk in {time.perf_counter() - st:.0f} s", flush=True)
  streamer = Streamer(model, disks, m)
  # EVAL=corpus.jsonl[:nseq[:max_out]]: production completions teacher forced, each from position 0: per output token the production
  # token's logprob, top-1 agreement with production and KL(production top-5 + the rest || ours)
  if (eval_spec:=getenv("EVAL", "")):
    from tinygrad import TinyJit
    class Keep:
      def __init__(self, lin): self.lin = lin
      def __call__(self, x):
        self.out = self.lin(x)
        return self.out
    model.output = Keep(model.output)  # type: ignore[assignment]
    ejits = {T: TinyJit(lambda *a: (model.forward(*a), model.output.out.float().contiguous().to(Device.DEFAULT))) for T in (1, streamer.T)}  # type: ignore[attr-defined]
    for jit in ejits.values(): jit.cnt = 1
    logits:list = []
    def eval_model(tokens, *a):
      tok, lg = ejits[tokens.shape[1]](tokens, *a)
      logits.append(lg)
      return tok
    streamer.model = eval_model  # type: ignore[assignment]

  model.rollout_jit.cnt = 1
  # the rng counters now: the random inits of the parameters (replaced by the weights) left ~2000 increments pending, the graph's sampling
  # would take them in and run them every step
  Tensor.realize(*Tensor._device_rng_counters.values())
  print(f"loaded in {time.perf_counter() - st:.0f} s", flush=True)
  v_pos, temp = UOp.variable("start_pos", 0, model.max_context - 1), Tensor([0.0])
  if eval_spec:
    import json
    path, nseq, max_out = (eval_spec.split(":") + ["4", "256"])[:3]
    # (VERIFY=T: verify steps of T teacher tokens, every position's logits scored: the quality of the speculative path)
    stats, steps, T = [], 0, max(streamer.T, 1)
    model.verify = T > 1
    st = time.perf_counter()
    for r in [json.loads(line) for line in open(path)][:int(nseq)]:
      ids, outs = r["prompt_ids"], r["output_ids"][:int(max_out)]
      seq = ids + outs
      for pos0 in range(0, len(seq) - 1, T):
        if (steps:=steps + 1) == 2: c0 = streamer.counters()
        logits.clear()
        streamer.steps(toks:=seq[pos0:min(pos0 + T, len(seq) - 1)], pos0, v_pos, temp)
        for pos, lg in zip(range(pos0, pos0 + len(toks)), logits[0].numpy().reshape(len(toks), -1).astype(np.float64)):
          if pos < len(ids) - 1: continue
          j = pos - len(ids) + 1  # this position predicts output token j
          lp = lg - (lg.max() + np.log(np.exp(lg - lg.max()).sum()))
          top = r["output_top"][j]
          p = np.exp([x[0] for x in top])
          q = lp[[x[1] for x in top]]
          rest_p, rest_q = max(1 - p.sum(), 1e-12), max(1 - np.exp(q).sum(), 1e-12)
          kl = float((p * (np.log(p) - q)).sum() + rest_p * (np.log(rest_p) - np.log(rest_q)))
          stats.append((lp[outs[j]], r["output_logprobs"][j], float(int(lp.argmax()) == top[0][1]), kl))
      a_ = np.array(stats)
      print(f"{r['task_id']}: {len(stats)} tokens, ours logprob {a_[:, 0].mean():.4f} (production {a_[:, 1].mean():.4f}), "
            f"top-1 agree {a_[:, 2].mean():.1%}, KL top-5 {a_[:, 3].mean():.4f}, {time.perf_counter() - st:.0f} s", flush=True)
      streamer.report(f"tau {TAU} prefetch {PREFETCH}", c0, steps - 1)
    sys.exit(0)
  ids = ([tok.bos_id] if tok.bos_id is not None else []) + tok.encode(sys.argv[2] if len(sys.argv) > 2 else "The capital of France is")
  def chunk_prep(T:int): pass  # (a chunk size's graphs and buffers before its first step)
  if (draft_dir:=getenv("DRAFT", "")):
    # DRAFT=dir: speculative decoding of N tokens (VERIFY=2, SLOTS 6700 by default). the draft guesses the token after the next from the
    # target's hidden states after blocks 1, 45, 89, in the step's graph: a verify step of [next, guess] gives the target's token after
    # each and the draft's guess after each of those. the guess right, both and the one after are in (the next guess the second's), else
    # only the next (the first's) and the recurrent states go back to after it. (the draft at the first position is the same either way:
    # the target's next token on the hidden states after the step's first; past the end of the accepted its cache is overwritten next)
    from tinygrad import TinyJit
    from extra.k3.draft import Eagle3Draft
    model.verify, model.keep_hidden = True, (1, 45, 89)
    draft = Eagle3Draft(draft_dir, lambda t: model.token_embd(t.to(model.token_embd.weight.device)).float(), devs, model.max_context)
    # the step's tokens and hidden states into buffers the draft reads (its own jit, launched right after the layers: no sync between)
    toks_out = {T: Tensor.empty(1, T, dtype=dtypes.int32, device=Device.DEFAULT).realize() for T in (1, 2)}
    hidden = {T: Tensor.empty(1, T, 3, model.config.dim, device=devs).realize() for T in (1, 2)}
    def tapped(tokens:Tensor, *a) -> Tensor:
      out, T = model.forward(tokens, *a), int(tokens.shape[1])
      Tensor.realize(Tensor(hidden[T].uop).assign(Tensor.stack(*[model.hidden[i][:, :, -1] for i in model.keep_hidden], dim=2).float()),
                     tok:=Tensor(toks_out[T].uop).assign(out.cast(dtypes.int32)))
      return tok
    target = {T: TinyJit(tapped) for T in (1, 2)}
    for tj in target.values(): tj.cnt = 1
    streamer.model = lambda tokens, *a: target[tokens.shape[1]](tokens, *a)  # type: ignore[assignment]
    given = host_tensor(Device.DEFAULT, (1, 1), dtypes.int32)  # (a prompt's step: the draft's token, its next prompt token; -1 the target's)
    def guesses(T:int, p:UOp) -> Tensor:
      dtoks = toks_out[T] if T > 1 else (given[0] >= 0).where(given[0], toks_out[T])
      return draft(dtoks, hidden[T], p).to(Device.DEFAULT).argmax(-1).cast(dtypes.int32)
    drafts = {T: TinyJit(functools.partial(guesses, T)) for T in (1, 2)}
    # a prompt's chunk: the target's hidden states of every token (not verify: the next token after the last only), the draft's cache of
    # the chunk's positions on the prompt's tokens after them (given_c)
    given_c:dict[int, tuple[Tensor, np.ndarray]] = {}
    def chunk_jits(T:int):
      if T in given_c: return
      given_c[T], hidden[T] = host_tensor(Device.DEFAULT, (1, T), dtypes.int32), Tensor.empty(1, T, 3, model.config.dim, device=devs).realize()
      def tapped_chunk(tokens:Tensor, *a) -> Tensor:
        model.verify = False
        out, model.verify = model.forward(tokens, *a), True
        Tensor.realize(Tensor(hidden[T].uop).assign(Tensor.stack(*[model.hidden[i][:, :, -1] for i in model.keep_hidden], dim=2).float()))
        return out
      target[T], drafts[T] = TinyJit(tapped_chunk), TinyJit(lambda p: draft(given_c[T][0], hidden[T], p, head=False))
      target[T].cnt = drafts[T].cnt = 1
    streamer.chunk_model, chunk_prep = lambda tokens, *a: target[tokens.shape[1]](tokens, *a), chunk_jits  # type: ignore[assignment]
    streamer.after = lambda T, p: drafts[T](p)
  else:
    from tinygrad import TinyJit
    plain:dict[int, Callable[..., Tensor]] = {}
    def plain_chunk(tokens:Tensor, *a) -> Tensor:
      if (T:=int(tokens.shape[1])) not in plain:
        plain[T] = jit = TinyJit(model.forward)
        jit.cnt = 1
      return plain[T](tokens, *a)
    streamer.chunk_model = plain_chunk
  def decode(ids:list[int], pos:int, n:int, emit:Callable[[int], None]=lambda t: None) -> tuple[list[int], int, list, tuple]:
    """the greedy tokens after ids (fed from position pos), up to n or an end token (not fed), each to emit as it comes: them, the
    position after the last fed, each step's (right, step, rollback) times and (counters, tokens, steps) once every jit's captured"""
    i = 0  # the prompt but its last token in chunks (CHUNKS, the largest that fits first), the rest a token a time (the draft at every position)
    while (C:=next((c for c in CHUNKS if len(ids) - 1 - i >= c), 0)):
      t0 = time.perf_counter()
      chunk_prep(C)
      if draft_dir: given_c[C][1][0, :] = ids[i + 1:i + C + 1]
      streamer.chunk(ids[i:i + C], pos + i, v_pos, temp)
      i += C
      if getenv("VCHUNK"): print(f"chunk of {C} at {pos + i - C}: {time.perf_counter() - t0:.2f} s", flush=True)
    for i, t in enumerate(ids[i:], i):
      t0 = time.perf_counter()
      if draft_dir: given[1][0, 0] = ids[i + 1] if i + 1 < len(ids) else -1
      nxt, *g = streamer.steps([t], pos + i, v_pos, temp)
      if pos + i == 0: print(f"first step in {time.perf_counter() - t0:.0f} s", flush=True)
    pos, out, times, warm = pos + len(ids), [nxt], [], None
    emit(nxt)
    while len(out) < n and (DRY or not tok.is_end(out[-1])):
      t0, o = time.perf_counter(), len(out)
      if not draft_dir:
        out, pos, right = out + [streamer.step(out[-1], pos, v_pos, temp)], pos + 1, True
      elif (right:=(r:=streamer.steps([out[-1], g[0]], pos, v_pos, temp))[0] == g[0]) and not (tok.is_end(g[0]) and not DRY):
        out, g, pos = out + r[:2], r[3:], pos + 2
      else:  # (a right guess of the end: the end isn't fed either)
        model.rollback(1)
        out, g, pos = out + ([g[0]] if right else r[:1]), r[2:3], pos + 1
      t1 = time.perf_counter()
      for t in out[o:]: emit(t)
      times.append((right, t1 - t0, time.perf_counter() - t1))
      # (counted once every jit's captured: two right and two wrong guesses, two plain steps. DRY: the guesses are garbage, eight steps)
      if warm is None and (len(times) == (8 if draft_dir else 2) if not draft_dir or DRY else
                           min(sum(x[0] for x in times), sum(not x[0] for x in times)) == 2):
        warm = (streamer.counters(), len(out), len(times))
    return out, pos, times, warm
  if getenv("CHAT") or getenv("BENCH", ""):
    # CHAT=1: a chat in the model's template, THINK=off|low|high|max. the caches only go forward: a turn feeds the end of the last reply
    # (its end token isn't fed) and its own message, without the template's system prefix
    # BENCH=in.jsonl:out.jsonl: each line's "messages" answered from position 0 (up to MAXTOK tokens), a line out each: its "id", the reply
    # (the ids already out skipped: a run picks up where the last stopped)
    import json, jinja2
    env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda obj, **kw: json.dumps(obj, **kw)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
    env.globals["strftime_now"] = time.strftime
    template, think = env.from_string(kv["tokenizer.chat_template"]), getenv("THINK", "off")
    kw:dict = {"thinking": False} if think == "off" else {"thinking": True, "thinking_effort": think}
    prefix = template.render(messages=[], add_generation_prompt=False, **kw)
    fed = 0
    # the template's tags (<|open|>name<|sep|>, <|close|>name<|sep|>) aren't shown: from an open or close to the sep
    tag_open, tag_close, tag_sep = (tok.encode(t)[0] for t in ("<|open|>", "<|close|>", "<|sep|>"))
    # every step's jit captured before the first turn (its tokens overwritten: the recurrent states start again at position 0)
    decode(tok.encode(template.render(messages=[{"role": "user", "content": "Write a poem about the sea."}], add_generation_prompt=True, **kw)),
           0, 64)
    print(f"warm in {time.perf_counter() - st:.0f} s", flush=True)
    if (bench:=getenv("BENCH", "")):
      src, dst = bench.split(":")
      done = {json.loads(line)["id"] for line in open(dst)} if os.path.exists(dst) else set()
      with open(dst, "a") as f:
        for r in [r for line in open(src) if (r:=json.loads(line))["id"] not in done]:
          t0, ids = time.perf_counter(), tok.encode(template.render(messages=r["messages"], add_generation_prompt=True, **kw))
          out, _, times, _ = decode(ids, 0, min(getenv("MAXTOK", 2048), model.max_context - 2 - len(ids)))
          ntok, t_dec = len(out) - tok.is_end(out[-1]), sum(x[1] + x[2] for x in times)
          f.write(json.dumps({"id": r["id"], "reply": tok.decode(out[:ntok]), "tokens": ntok, "tok_s": ntok / max(t_dec, 1e-9),
                              "s": time.perf_counter() - t0}) + "\n")
          f.flush()
          print(f"{r['id']}: {ntok} tokens, {ntok / max(t_dec, 1e-9):.1f} tok/s", flush=True)
      sys.exit(0)
    if sys.stdin.isatty():  # what was typed while it loaded isn't a message
      import termios
      termios.tcflush(sys.stdin, termios.TCIFLUSH)
    while True:
      try: msg = input(">>> ").strip()
      except EOFError: break
      if not msg: continue
      turn = template.render(messages=[{"role": "user", "content": msg}], add_generation_prompt=True, **kw)
      ids = tok.encode(turn) if not fed else tok.encode(tok.decode([tok.eos_id]) + turn[len(prefix):])
      if fed + len(ids) >= model.max_context - 2:
        print(f"[context full: {fed}+{len(ids)} of {model.max_context} tokens, start again with a larger CTX]")
        break
      dec, shown = tok.stream_decoder(), [""]
      def show(tk:int):
        if tok.is_end(tk): return
        if tk in (tag_open, tag_close): shown[0] = " "
        elif tk == tag_sep:
          if shown[0].strip() == "response" and think != "off": sys.stdout.write("\n\n")  # the answer after the thinking
          shown[0] = ""
        elif shown[0]: shown[0] += dec(tk)
        else: sys.stdout.write(dec(tk))
        sys.stdout.flush()
      out, fed, times, _ = decode(ids, fed, min(getenv("MAXTOK", 2048), model.max_context - 2 - fed - len(ids)), show)
      ntok, t_turn = len(out) - tok.is_end(out[-1]), sum(x[1] + x[2] for x in times)
      print(dec() + f"\n[{ntok} tokens, {ntok / max(t_turn, 1e-9):.1f} tok/s, context {fed}/{model.max_context}]\n")
    sys.exit(0)
  # N tokens decoded after the prompt
  out, _, times, (c0, o0, m0) = decode(ids, 0, getenv("N", 32))
  if not DRY: print(repr(tok.decode(out)), out, flush=True)  # DRY=1 slots are garbage, so are the tokens
  if not draft_dir: streamer.report("dry" if DRY else f"tau {TAU} prefetch {PREFETCH} half life {HALF_LIFE}", c0, len(out) - o0)
  else:
    r, tv, td = (np.array(x) for x in zip(*times[m0:]))
    streamer.report(f"speculative, {r.mean():.1%} of {len(r)} guesses right, {(len(out) - o0) / len(r):.2f} tokens a step, verify step "
                    f"{np.median(tv) * 1e3:.1f} ms (with the draft), rollback {np.median(td[~r]) * 1e3:.2f} ms", c0, len(out) - o0)
