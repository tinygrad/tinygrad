import random
from tinygrad import Device
from tinygrad.helpers import getenv, DEBUG

def main():
  seed = getenv("SEED", 1337)
  n_gpus = getenv("GPUS", 3)
  iters = getenv("ITERS", 10000000)

  print(f"{n_gpus} GPUs for {iters} iterations")
  devs = tuple([Device[f"{Device.DEFAULT}:{x}"] for x in range(n_gpus)])

  for i in range(iters):
    dev = random.choice(devs)
    q_t = random.choice([dev.hw_copy_queue_t, dev.hw_compute_queue_t])
    
    deps_sigs = random.randint(0, len(devs))
    wait_devs = random.sample(devs, deps_sigs)

    q = q_t()
    for d in wait_devs: q.wait(d.timeline_signal, d.timeline_value - 1)
    q.signal(dev.timeline_signal, dev.timeline_value).submit(dev)
    dev.timeline_value += 1

    if sync:=random.randint(0, 10) < 3: dev.synchronize()
    if DEBUG >= 2: print(f"{i}: {dev.device_id} timeline {dev.timeline_value}, wait for {[d.device_id for d in wait_devs]}, {sync=}")
    elif i % 100 == 0: print(f"\rCompleted {i} iterations", end='')

if __name__ == "__main__":
  main()
