#!/usr/bin/env python3
"""Two-device 1024 GEMM: row-partitioned FP16 inputs, true FP32 MAD accumulation, full oracle."""
import json, os, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


N = 1024
REMOTE_REPO = os.getenv("REMOTE_REPO", "/data/openpilot/tinygrad_repo")


def worker() -> None:
  from tinygrad import Device, dtypes
  from tinygrad.device import Buffer
  from extra.gemm import qcom_intensity_gemm as q
  from extra.gemm.ir3asm import get_envelope, inject

  first, rows, seed = int(os.environ["ROW_START"]), int(os.environ["ROWS"]), int(os.getenv("SEED", "1001"))
  if rows != N//2 or first not in (0, N//2): raise ValueError("worker partition must be one half of N=1024")
  rng = np.random.default_rng(seed)
  a_full = (rng.standard_normal((N, N), dtype=np.float32)*np.float32(1/32)).astype(np.float16)
  b_np = (rng.standard_normal((N, N), dtype=np.float32)*np.float32(1/32)).astype(np.float16)
  a_np = np.ascontiguousarray(a_full[first:first+rows])
  q.M, q.N, q.K, q.K4 = rows, N, N, N//4
  dev = Device["QCOM"]
  env, io, sz, ro = get_envelope(dev, q.make_direct_image_donor_src(2, 64))
  shader, hregs, fregs, loop_instrs = q.build_4x8_fp32_rotate_shader(
    dev, 64, k_count=N//4, k_unroll=3)
  lib = inject(env, io, sz, ro, shader, fregs=fregs, hregs=hregs, mergedregs=False)

  def upload(x: np.ndarray, dtype):
    ret = Buffer("QCOM", x.size, dtype).allocate()
    ret.copyin(memoryview(x).cast("B"))
    return ret

  a, b = upload(a_np, dtypes.half), upload(b_np, dtypes.half)
  c = Buffer("QCOM", rows*N, dtypes.float).allocate()
  prg = dev.runtime("gemm_h", lib, buf_dtypes=[
    ((0, dtypes.float, (rows, N//4, 4)),), ((0, dtypes.half, (rows, N//4, 4)),),
    ((1, dtypes.half, (N, N//4, 4)),)])
  for _ in range(2):
    prg(c._buf, a._buf, b._buf, global_size=(N//256, rows//8, 1), local_size=(64, 1, 1), wait=True)
  if start_ns := int(os.getenv("START_TIME_NS", "0")):
    delay = (start_ns-time.time_ns())/1e9
    if delay > 0: time.sleep(delay)
  times = [prg(c._buf, a._buf, b._buf, global_size=(N//256, rows//8, 1),
               local_size=(64, 1, 1), wait=True) for _ in range(int(os.getenv("BENCH_RUNS", "20")))]
  got = np.empty((rows, N), np.float32); c.copyout(memoryview(got).cast("B"))
  expected = a_np.astype(np.float32) @ b_np.astype(np.float32)
  delta = np.abs(got-expected)
  bad = ~np.isclose(got, expected, rtol=1e-3, atol=8e-3)
  output = Path(os.getenv("PART_OUTPUT", f"/tmp/qcom_fp32_rows_{first}.npy"))
  np.save(output, got)
  result = {"first": first, "rows": rows, "elapsed": min(times), "times": times, "bad_count": int(bad.sum()),
            "max_abs": float(delta.max()), "mean_abs": float(delta.mean()), "fregs": fregs,
            "loop_instrs": loop_instrs, "output": str(output)}
  print("RESULT_JSON="+json.dumps(result, sort_keys=True))
  if bad.any(): raise SystemExit(1)


def coordinator() -> None:
  hosts = os.getenv("QCOM_HOSTS", "tc3,tc4").split(",")
  if len(hosts) != 2: raise ValueError("QCOM_HOSTS must name exactly two devices")
  ssh_opts = ["-o", "ConnectTimeout=8", "-o", "BatchMode=yes"]
  here = Path(__file__).resolve()
  kernel = here.with_name("qcom_intensity_gemm.py")
  for host in hosts:
    subprocess.run(["scp", "-q", *ssh_opts, str(here), str(kernel), f"{host}:{REMOTE_REPO}/extra/gemm/"], check=True,
                   timeout=15)
  start_ns = time.time_ns()+15_000_000_000

  def launch(item: tuple[str, int]) -> tuple[str, dict]:
    host, first = item
    remote_output = f"/tmp/qcom_fp32_rows_{first}.npy"
    cmd = (f"cd {REMOTE_REPO} && PYTHONPATH=. DEV=QCOM IMAGE=1 FLOAT16=1 HCQ2=1 MODE=worker "
           f"ROW_START={first} ROWS={N//2} SEED={int(os.getenv('SEED', '1001'))} PART_OUTPUT={remote_output} "
           f"START_TIME_NS={start_ns} BENCH_RUNS={int(os.getenv('BENCH_RUNS', '20'))} "
           f".venv/bin/python extra/gemm/{here.name}")
    done = subprocess.run(["ssh", *ssh_opts, host, cmd], check=True, text=True, capture_output=True, timeout=60)
    line = next(x for x in done.stdout.splitlines() if x.startswith("RESULT_JSON="))
    result = json.loads(line.removeprefix("RESULT_JSON="))
    return host, result

  with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(launch, zip(hosts, (0, N//2))))
  with tempfile.TemporaryDirectory() as tmp:
    parts = []
    for host, result in results:
      local = Path(tmp)/f"part_{result['first']}.npy"
      subprocess.run(["scp", "-q", *ssh_opts, f"{host}:{result['output']}", str(local)], check=True, timeout=15)
      parts.append((result["first"], np.load(local), result))
    parts.sort()
    got = np.concatenate([x[1] for x in parts])
  rng = np.random.default_rng(int(os.getenv("SEED", "1001")))
  a_np = (rng.standard_normal((N, N), dtype=np.float32)*np.float32(1/32)).astype(np.float16)
  b_np = (rng.standard_normal((N, N), dtype=np.float32)*np.float32(1/32)).astype(np.float16)
  expected = a_np.astype(np.float32) @ b_np.astype(np.float32)
  delta = np.abs(got-expected); bad = ~np.isclose(got, expected, rtol=1e-3, atol=8e-3)
  paired = [max(parts[0][2]["times"][i], parts[1][2]["times"][i]) for i in range(len(parts[0][2]["times"]))]
  best_i = int(np.argmin(paired)); elapsed = paired[best_i]
  print(f"shape={N}x{N}x{N} devices={','.join(hosts)} inputs=fp16 accumulate=fp32 elapsed_ms={elapsed*1e3:.3f} "
        f"gflops={2*N**3/elapsed/1e9:.1f} outputs={N*N} bad_count={int(bad.sum())} "
        f"max_abs={float(delta.max()):.9g} mean_abs={float(delta.mean()):.9g} allclose={not bool(bad.any())} "
        f"part_ms={[round(x[2]['times'][best_i]*1e3, 3) for x in parts]} paired_iteration={best_i}")
  if bad.any() or 2*N**3/elapsed/1e9 <= 400: raise SystemExit(1)


if __name__ == "__main__":
  worker() if os.getenv("MODE") == "worker" else coordinator()
