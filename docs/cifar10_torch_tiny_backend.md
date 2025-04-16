# How to Use `tinygrad` as a Backend for `pytorch-cifar` Training

This guide explains how to configure and run the `pytorch-cifar` training example using `tinygrad` as the backend, after installing `tinygrad`.

## 1. Clone the Repository

First, clone the `pytorch-cifar` repository:

```shell
git clone https://github.com/kuangliu/pytorch-cifar.git
cd pytorch-cifar # Navigate into the directory
```

## 2. Modify `main.py`

Next, modify the `main.py` file within the cloned repository to enable the `tinygrad` backend. Apply the following changes:

```diff
 import torch.optim as optim
 import torch.nn.functional as F
 import torch.backends.cudnn as cudnn
+from tinygrad import getenv

 import torchvision
 import torchvision.transforms as transforms
@@ -21,7 +22,12 @@ parser.add_argument('--resume', '-r', action='store_true',
                     help='resume from checkpoint')
 args = parser.parse_args()

-device = 'cuda' if torch.cuda.is_available() else 'cpu'
+if getenv("TINY_BACKEND"):
+    import tinygrad.frontend.torch
+    device = torch.device("tiny")
+else:
+    device = 'cuda' if torch.cuda.is_available() else 'cpu'
+
 best_acc = 0  # best test accuracy
 start_epoch = 0  # start from epoch 0 or last checkpoint epoch

@@ -55,7 +61,8 @@ classes = ('plane', 'car', 'bird', 'cat', 'deer',
 # Model
 print('==> Building model..')
 # net = VGG('VGG19')
-# net = ResNet18()
+# Use ResNet18 for this example
+net = ResNet18()
 # net = PreActResNet18()
 # net = GoogLeNet()
 # net = DenseNet121()
@@ -68,7 +75,7 @@ print('==> Building model..')
 # net = ShuffleNetV2(1)
 # net = EfficientNetB0()
 # net = RegNetX_200MF()
-net = SimpleDLA()
+# net = SimpleDLA() # Comment out SimpleDLA if you uncommented ResNet18
 net = net.to(device)
 if device == 'cuda':
     net = torch.nn.DataParallel(net)
```

**Note:** This diff enables `ResNet18`. Ensure other models like `SimpleDLA` are commented out.

## 3. Run the Training

Execute the training script using the following command. Make sure to replace `path-to-tinygrad` with the actual path to your `tinygrad` installation if it's not in the standard Python path.

```shell
PYTHONPATH=path-to-tinygrad TINY_BACKEND=1 python3 main.py
```

## 4. Expected Output

You should observe output similar to the following, indicating the training progress with the `tinygrad` backend:

```text
==> Preparing data..
==> Building model..

Epoch: 0
/workspace/.python/tinygrad/lib/python3.10/site-packages/torch/nn/modules/module.py:1830: FutureWarning: Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated and will be removed in future versions. This hook will be missing some grad_input. Please use register_full_backward_hook to get the documented behavior.
  self._maybe_warn_non_full_backward_hook(args, result, grad_fn)
 [=========================== 391/391 ============================>]  Step: 20s974ms | Tot: 9m27s | Loss: 1.990 | Acc: 29.360% (14680/50000)
 [=========================== 100/100 ============================>]  Step: 204ms | Tot: 20s660ms | Loss: 1.688 | Acc: 40.040% (4004/10000)
Saving..

Epoch: 1
 [=========================== 391/391 ============================>]  Step: 1s298ms | Tot: 9m17s | Loss: 1.483 | Acc: 45.402% (22701/50000)
 [=========================== 100/100 ============================>]  Step: 209ms | Tot: 21s330ms | Loss: 1.289 | Acc: 52.780% (5278/10000)
Saving..
...
```
```

This revised version uses headings (`##`) to structure the steps, specifies the language for code blocks (`shell`, `diff`, `text`), adds a `cd` command after cloning, and includes minor clarifications.
