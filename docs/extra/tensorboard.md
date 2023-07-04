# tinygrad Tensorboard guide

## Installation
```shell
pip install tensorboard
```

### For ARM-based Mac users
There is an [issue](https://github.com/tensorflow/io/issues/1625) with a dependency for tensorboard, `tensorboard-io`, which is not available for ARM-based Macs. 
To get around this, you can build the dependency from source:
```shell
git clone https://github.com/tensorflow/io
cd io
python setup.py bdist_wheel
cd dist
python -m pip install --no-deps tensorflow_io-0.32.0-cp310-cp310-macosx_11_0_arm64.whl
```

## Usage
Start the tensorboard server:
```shell
tensorboard --logdir runs
```
Then, in your python script, import the `TinySummaryWriter` class to write to the logs dir.
```python
import numpy as np
from extra.tensorboard.writer import TinySummaryWriter

if __name__ == "__main__":
    writer = TinySummaryWriter() # default log_dir is runs
    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
    writer.close()
```
Then, open your browser to `localhost:6006` to view the results.
