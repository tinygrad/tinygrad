import torch
import torch.nn as nn


#torch.manual_seed(42)
# Create a simple model that outputs logits
model = nn.Linear(10, 1)


model.weight.data = torch.tensor([[ 0.2144, -0.0718, -0.1301,  0.1295,  0.2638,  0.2497, -0.0458, -0.2414,
          0.0608,  0.2978]])
# Create a BCEWithLogitsLoss instance
model.bias.data = torch.tensor([0.2])


print(f"pytorch model weights: {model.weight} \n")
print(f"pytorch model bias: {model.bias} \n \n")



criterion = nn.BCEWithLogitsLoss()

criterion2 = nn.BCELoss()
# Create some example data
inputs = torch.tensor([[-0.9342, -0.2483, -1.2082, -0.4777,  0.5201,  1.6423, -0.1596, -0.4974,
         -0.9634,  2.0024],
        [ 0.4664,  1.5730, -0.9228,  1.2791,  0.3211,  1.5736, -0.8455,  1.3123,
          0.6872, -1.0892],
        [-0.3553, -0.9138,  0.8963,  0.0499,  2.2667,  1.1790, -0.4345, -1.3864,
         -1.2862, -1.4032]])  # batch size of 3, 10 features
labels = torch.tensor([0, 1, 0]).float().view(-1, 1)  # true labels

# Forward pass through the model
outputs = model(inputs)

# Compute the loss
loss = criterion(outputs, labels)
loss2 = criterion2(outputs, labels)
print(f"pytorch BCEWithLogitsLoss loss: {loss}")
print(f"pytorch BCEloss: {loss2}")
#print(inputs)



from extra.training import *
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.nn import *
from tinygrad.tensor import *
from tinygrad.tensor import Sequence

tiny_model = Linear(10,1)
tiny_model.weight = Tensor([[ 0.2144, -0.0718, -0.1301,  0.1295,  0.2638,  0.2497, -0.0458, -0.2414,
          0.0608,  0.2978]])
tiny_model.bias = Tensor([0.2])

#print(tiny_model.weight.numpy())
#print(tiny_model.bias.numpy())

print(f"tinygrad model weights: {tiny_model.weight.numpy()} \n")
print(f"tinygrad model bias: {tiny_model.bias.numpy()} \n \n")

tiny_criterion = BCEWithLogitsLoss
tiny_criterion2 = BCELoss

#numpy_array = array
#test_tensor
tiny_inputs = Tensor([[-0.9342, -0.2483, -1.2082, -0.4777,  0.5201,  1.6423, -0.1596, -0.4974,
         -0.9634,  2.0024],
        [ 0.4664,  1.5730, -0.9228,  1.2791,  0.3211,  1.5736, -0.8455,  1.3123,
          0.6872, -1.0892],
        [-0.3553, -0.9138,  0.8963,  0.0499,  2.2667,  1.1790, -0.4345, -1.3864,
         -1.2862, -1.4032]])

tiny_labels = Tensor([0,1,0])

tiny_outputs = tiny_model(tiny_inputs)

tiny_loss = tiny_criterion(tiny_outputs, tiny_labels)
tiny_loss2 = tiny_criterion(tiny_outputs, tiny_labels)

print(f"tinygrad BCEWithLogitsLoss : {tiny_loss.numpy()}")
print(f"tinygrad BCELoss: {tiny_loss2.numpy()}")
#print(tiny_inputs.numpy())
