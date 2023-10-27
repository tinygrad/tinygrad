import tinygrad as tg
from tinygrad.tensor import Tensor
from models.resnet import ResNet
import numpy as np
import gzip
from tinygrad.helpers import dtypes
import tinygrad.nn.optim as optim
from tinygrad.nn.state import get_parameters

# Dataset source: https://github.com/golbin/TensorFlow-MNIST/blob/master/mnist/data/t10k-labels-idx1-ubyte.gz

train_X_file = "train-images-idx3-ubyte.gz"
train_labels_file = "train-labels-idx1-ubyte.gz"
test_X_file = "t10k-images-idx3-ubyte.gz"
test_labels_file = "t10k-labels-idx1-ubyte.gz"

load_images = lambda path:np.frombuffer(gzip.open(path).read(), dtype=np.uint8).copy()[0x10:].reshape((-1, 28, 28)).astype(np.float32)/255
load_labels = lambda path:np.frombuffer(gzip.open(path).read(), dtype=np.uint8).copy()[8:]

X_train = load_images(train_X_file)
y_train = load_labels(train_labels_file)
X_test = load_images(test_X_file)
y_test = load_labels(test_labels_file)
# Convert to RGB and convert to Tensors
X_train = Tensor(np.stack((X_train,)*3, axis=-1).reshape((-1,3,28,28)))
X_test = Tensor(np.stack((X_test,)*3, axis=-1).reshape((-1,3,28,28)))
y_train = Tensor(y_train)
y_test = Tensor(y_test)

model = ResNet(18,num_classes=10)
optimizer = optim.Adam(get_parameters(model),lr=1e-4)

NUM_EPOCHS = 10
batch_size = 32

def sparse_categorical_crossentropy(self, Y, ignore_index=-1) -> Tensor:
    loss_mask = Y != ignore_index
    y_counter = Tensor.arange(self.shape[-1], dtype=dtypes.int32, requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), self.shape[-1])
    y = ((y_counter == Y.flatten().reshape(-1, 1)).where(-1.0, 0) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
    return self.log_softmax().mul(y).sum() / loss_mask.sum()


for epoch in range(NUM_EPOCHS):
    for batch in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[batch:batch+batch_size]
        y_batch = y_train[batch:batch+batch_size]
        out = model.forward(X_batch)
        loss = sparse_categorical_crossentropy(out,y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()       
        pred = out.argmax(axis=-1)
        accuracy = (pred == y_batch).mean()
        print(f"Loss: {loss.numpy()} | Accuracy: {accuracy.numpy()}")
