#!/usr/bin/env python3
# runs where the quantized K3 gguf lives: streams the routed experts, cut into per-gpu parts, and the trunk gguf to extra/k3/recv.py.
# python3 pack.py experts.gguf trunk.gguf host:port [--units 1024,512,512,1024] (trunk "-": only the experts)
import argparse, json, os, socket, struct, sys
import numpy as np

ap = argparse.ArgumentParser()
for a in ("experts", "trunk", "dest"): ap.add_argument(a)
ap.add_argument("--units", default="1024,512,512,1024", help="intermediate units of each expert per gpu, multiples of the quant block")
ap.add_argument("--gguf_py", default="/home/nimlgen/k3q/atomic-llama/gguf-py")
args = ap.parse_args()
sys.path.insert(0, args.gguf_py)
import gguf

r = gguf.GGUFReader(args.experts)
t = {x.name: x for x in r.tensors}
layers = sorted({int(n.split(".")[1]) for n in t if n.endswith("ffn_gate_exps.weight")})
g0 = t[f"blk.{layers[0]}.ffn_gate_exps.weight"]
qtype = g0.tensor_type
blk, tsz = gguf.GGML_QUANT_SIZES[qtype]
n_experts, ff, lat = (int(x) for x in g0.shape[::-1])  # gguf shape is [in, out, experts]
units = [int(u) for u in args.units.split(",")]
assert sum(units) == ff and all(u % blk == 0 for u in units), f"units {units} must split {ff} in multiples of {blk}"
starts = [sum(units[:i]) for i in range(len(units))]
row_up, row_down = lat // blk * tsz, ff // blk * tsz  # bytes of a gate/up row (lat inputs) and of a down row (ff inputs)
parts = [-(-(2 * u * row_up + lat * (u // blk * tsz)) // 4096) * 4096 for u in units]

trunk_size = os.path.getsize(args.trunk) if args.trunk != "-" else 0
header = {"model": "kimi-k3", "expert_type": qtype.name, "block": [blk, tsz], "units": units, "lat": lat, "ff": ff, "n_experts": n_experts,
          "layers": layers, "part_bytes": parts, "trunk_bytes": trunk_size, "trunk_name": os.path.basename(args.trunk) if trunk_size else ""}
host, port = args.dest.rsplit(":", 1)
sock = socket.create_connection((host, int(port)))
raw = json.dumps(header).encode()
sock.sendall(struct.pack("<Q", len(raw)) + raw)

for L in layers:
  gate, up, down = (np.asarray(t[f"blk.{L}.ffn_{n}_exps.weight"].data).reshape(n_experts, -1) for n in ("gate", "up", "down"))
  gate, up, down = gate.reshape(n_experts, ff, row_up), up.reshape(n_experts, ff, row_up), down.reshape(n_experts, lat, row_down)
  for e in range(n_experts):
    for u0, u, size in zip(starts, units, parts):
      b0, b1 = u0 // blk * tsz, (u0 + u) // blk * tsz
      part = np.concatenate([gate[e, u0:u0 + u].reshape(-1), up[e, u0:u0 + u].reshape(-1), down[e, :, b0:b1].reshape(-1)])
      sock.sendall(memoryview(part))
      sock.sendall(bytes(size - part.nbytes))
  print(f"layer {L} sent", flush=True)

if trunk_size:
  with open(args.trunk, "rb") as f:
    while (b:=f.read(64 << 20)): sock.sendall(b)
sock.shutdown(socket.SHUT_WR)
print("done:", sock.recv(16), flush=True)
