#!/usr/bin/env python3
# DEV=AMD python3 extra/k3/test_kda_decode.py: amd.kda_decode (a kda token in one kernel) against numpy, from a fresh and a running state
import numpy as np
from tinygrad import Tensor, UOp
from tinygrad.llm.kernels.amd import kda_decode, _q8_quantize, _q8_cache

H, D, LB, EPS = 4, 128, -5.0, 1e-5
rng = np.random.default_rng(0)
def rand(*shape, s=1.0): return (rng.standard_normal(shape) * s).astype(np.float32)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def silu(x): return x * sigmoid(x)

def ref(y, conv, conv_w, alpha_lin, dt_bias, ssm_a, state, norm_w, initial):
  qd = H * D
  if initial: conv, state = np.zeros_like(conv), np.zeros_like(state)
  win = np.concatenate([conv[0], y[:3 * qd].reshape(1, 3, qd)])  # (4, 3, qd)
  out = silu((win * conv_w.transpose(2, 0, 1)).sum(0))
  q, k, v = (out[i].reshape(H, D) for i in range(3))
  q, k = (x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6) for x in (q, k))
  q = q * D ** -0.5
  gate, beta = y[3 * qd:3 * qd + H * D].reshape(H, D), sigmoid(y[3 * qd + H * D:])
  alpha = np.exp(sigmoid((alpha_lin + dt_bias).reshape(H, D) * -ssm_a) * LB)
  s1 = state[0] * alpha[:, None, :]
  delta = (v - (s1 * k[:, None, :]).sum(-1)) * beta[:, None]
  new_state = s1 + delta[:, :, None] * k[:, None, :]
  core = (new_state * q[:, None, :]).sum(-1)
  z = core / np.sqrt((core * core).mean(-1, keepdims=True) + EPS) * norm_w * sigmoid(gate)
  return z.reshape(-1).astype(np.float16), win[1:][None], new_state[None]

for pos in (0, 5):
  y, conv, conv_w = rand(3 * H * D + H * D + H), rand(1, 3, 3, H * D), rand(3, H * D, 4, s=0.5)
  alpha_lin, dt_bias, ssm_a, state, norm_w = rand(H * D), rand(H * D), np.abs(rand(H, 1)), rand(1, H, D, D, s=0.1), rand(D)
  want_z, want_conv, want_state = ref(y, conv, conv_w, alpha_lin, dt_bias, ssm_a, state, norm_w, pos == 0)
  t_conv, t_state = Tensor(conv).contiguous().realize(), Tensor(state).contiguous().realize()
  start_pos = UOp.variable("start_pos", 0, 1023).bind(pos)
  qd = H * D
  q, k, v, gate, beta = (Tensor(y[a:b]).reshape(1, 1, -1) for a, b in ((0, qd), (qd, 2*qd), (2*qd, 3*qd), (3*qd, 4*qd), (4*qd, 4*qd + H)))
  z, _, _ = kda_decode(q, k, v, gate, beta, t_conv, Tensor(conv_w), Tensor(alpha_lin).reshape(1, 1, -1), Tensor(dt_bias), Tensor(ssm_a),
                       t_state, Tensor(norm_w), start_pos, LB, EPS)
  q, scale, xsum = _q8_cache[z.uop]
  Tensor.realize(z, q, scale, xsum)
  np.testing.assert_allclose(z.numpy().reshape(-1).astype(np.float32), want_z.astype(np.float32), rtol=2e-2, atol=2e-2)
  np.testing.assert_allclose(t_state.numpy(), want_state, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(t_conv.numpy(), want_conv, rtol=1e-6)
  q2, scale2, xsum2 = _q8_quantize(Tensor(z.numpy().reshape(1, -1)), tokens=1, in_features=H * D)
  np.testing.assert_equal(q.numpy(), q2.numpy())
  np.testing.assert_allclose(scale.numpy(), scale2.numpy(), rtol=1e-6)
  print(f"start_pos {pos}: ok, max |z - ref| {np.abs(z.numpy().reshape(-1).astype(np.float32) - want_z).max():.2e}")

# src: the conv window and state read from other buffers, the written ones get the new window and state, the read ones stay
y, conv, conv_w = rand(3 * H * D + H * D + H), rand(1, 3, 3, H * D), rand(3, H * D, 4, s=0.5)
alpha_lin, dt_bias, ssm_a, state, norm_w = rand(H * D), rand(H * D), np.abs(rand(H, 1)), rand(1, H, D, D, s=0.1), rand(D)
want_z, want_conv, want_state = ref(y, conv, conv_w, alpha_lin, dt_bias, ssm_a, state, norm_w, False)
s_conv, s_state = Tensor(conv).contiguous().realize(), Tensor(state).contiguous().realize()
o_conv, o_state = Tensor.zeros(*conv.shape).contiguous().realize(), Tensor.zeros(*state.shape).contiguous().realize()
qd = H * D
q, k, v, gate, beta = (Tensor(y[a:b]).reshape(1, 1, -1) for a, b in ((0, qd), (qd, 2*qd), (2*qd, 3*qd), (3*qd, 4*qd), (4*qd, 4*qd + H)))
z, w_conv, w_state = kda_decode(q, k, v, gate, beta, o_conv, Tensor(conv_w), Tensor(alpha_lin).reshape(1, 1, -1), Tensor(dt_bias),
                                Tensor(ssm_a), o_state, Tensor(norm_w), UOp.variable("start_pos", 0, 1023).bind(5), LB, EPS, src=(s_conv, s_state))
Tensor.realize(z, w_conv, w_state)
np.testing.assert_allclose(z.numpy().reshape(-1).astype(np.float32), want_z.astype(np.float32), rtol=2e-2, atol=2e-2)
np.testing.assert_allclose(o_state.numpy(), want_state, rtol=1e-4, atol=1e-4)
np.testing.assert_allclose(o_conv.numpy(), want_conv, rtol=1e-6)
np.testing.assert_equal(s_state.numpy(), state)
np.testing.assert_equal(s_conv.numpy(), conv)
print("src: ok, the read state unchanged")

# a step of 2 tokens: the kernel of each reads its row of the linears' outputs and writes its row of z and its q8, in place
ys = [rand(3 * H * D + H * D + H) for _ in range(2)]
conv, conv_w = rand(1, 3, 3, H * D), rand(3, H * D, 4, s=0.5)
alphas, dt_bias, ssm_a, state, norm_w = [rand(H * D) for _ in range(2)], rand(H * D), np.abs(rand(H, 1)), rand(1, H, D, D, s=0.1), rand(D)
want_zs, w_conv, w_state = [], conv, state
for y, al in zip(ys, alphas):
  wz, w_conv, w_state = ref(y, w_conv, conv_w, al, dt_bias, ssm_a, w_state, norm_w, False)
  want_zs.append(wz)
t_conv, t_state = Tensor(conv).contiguous().realize(), Tensor(state).contiguous().realize()
qd = H * D
lin = [Tensor(np.stack([y[a:b] for y in ys])).reshape(1, 2, -1) for a, b in ((0, qd), (qd, 2*qd), (2*qd, 3*qd), (3*qd, 4*qd), (4*qd, 4*qd + H))]
z, sp = None, UOp.variable("start_pos", 0, 1023).bind(5)
for t in range(2):
  z, t_conv, t_state = kda_decode(*lin, t_conv, Tensor(conv_w), Tensor(np.stack(alphas)).reshape(1, 2, -1), Tensor(dt_bias), Tensor(ssm_a), t_state,
                                  Tensor(norm_w), sp + t, LB, EPS, token=t, z=z)
assert z is not None
q, scale, xsum = _q8_cache[z.uop]
Tensor.realize(z, q, scale, xsum, t_conv, t_state)
np.testing.assert_allclose(z.numpy().reshape(2, -1).astype(np.float32), np.stack(want_zs).astype(np.float32), rtol=2e-2, atol=2e-2)
np.testing.assert_allclose(t_state.numpy(), w_state, rtol=1e-4, atol=1e-4)
np.testing.assert_allclose(t_conv.numpy(), w_conv, rtol=1e-6)
q2, scale2, xsum2 = _q8_quantize(Tensor(z.numpy().reshape(2, -1)), tokens=2, in_features=H * D)
np.testing.assert_equal(q.numpy(), q2.numpy())
np.testing.assert_allclose(scale.numpy(), scale2.numpy(), rtol=1e-6)
print("2 tokens: ok")
