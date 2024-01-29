from tinygrad.helpers import getenv
from tinygrad.device import Device

from tinygrad import Tensor

from train import fb_pass
import time


BS = getenv ("BS", 1)
GPUS = [f'{Device.DEFAULT}:{i}' for i in range(getenv("GPUS", 0))]

if GPUS: 
    assert BS % len(GPUS) == 0, "invalid BS"
    for x in GPUS: Device[x]


maxX, maxY = 501, 276


def timestring(s:int): return f"{int(s//3600)}:{int(s%3600//60)}:{s%60}"


for i in range(4):
    st = time.time()

    X_lens = Tensor.randint(BS, low = maxX/2, high = maxX-1).realize()
    Y_lens = Tensor.randint(BS, low = maxY/2, high = maxY-1).realize()

    X = Tensor.rand(maxX, BS, 240).realize()
    labels = Tensor.randint(BS, maxY, high = 29).realize()

    if GPUS:
        X.shard_(GPUS,axis=1)

        X_lens.shard_(GPUS,axis=0)
        Y_lens.shard_(GPUS,axis=0)
        labels.shard_(GPUS,axis=0)

    fb_pass(X,labels, X_lens, Y_lens,maxX, maxY)
    print (f"GPUS:{len(GPUS)} BS:{BS} run:{i} t:{timestring(time.time()-st)}",flush = True)

print()