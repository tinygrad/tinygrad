import torch, os
os.environ['TORCH_DEBUG'] = '1'

device = "tiny"


def dump_tensor(name, t):
  print(f"{name}: shape={tuple(t.shape)} device={t.device} requires_grad={t.requires_grad}")

print("=== SETUP ===")
x = torch.randn(4,4, device='tiny', requires_grad=True)
dump_tensor('x', x)

print("\n=== First slice: x[:, 1:2] ===")
y = x[:, 1:2]
dump_tensor('y', y)

print("\n=== Second slice (tuple indexing) y[:, 0:1] ===")
try:
  z = y[:, 0:1]
  dump_tensor('z', z)
  print("Tuple second slice WORKED (unexpected if failing before)")
except Exception as e:
  print(f"Tuple second slice FAILED: {e}")
  import traceback; traceback.print_exc()

print("\n=== Workaround slice: use narrow instead of tuple indexing ===")
try:
  y2 = x.narrow(1, 1, 1)            # first slice along dim 1
  dump_tensor('y2', y2)
  z2 = y2.narrow(1, 0, 1)           # second slice along dim 1 of view
  dump_tensor('z2', z2)
  print("narrow chain SUCCESS")
except Exception as e:
  print(f"narrow chain FAILED: {e}")

print("\n=== Gradient check on narrow chain ===")
try:
  (z2.sum()).backward()
  print("Backward on z2 succeeded. x.grad shape:", x.grad.shape)
except Exception as e:
  print(f"Backward failed: {e}")

print("\n=== Sequential single-dim slicing as workaround ===")
try:
  a1 = x[:, 1:2]
  a2 = a1[0:4]        # slice only one dimension (works) then simulate second by reshape + narrow
  dump_tensor('a1', a1)
  dump_tensor('a2', a2)
except Exception as e:
  print(f"Sequential workaround FAILED: {e}")

print("\nDone.")
