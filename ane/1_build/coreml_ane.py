#!/usr/bin/env python3
import numpy as np
import coremltools as ct
from coremltools.models.neural_network import datatypes, NeuralNetworkBuilder

input_features = [('image', datatypes.Array(3))]
output_features = [('probs', datatypes.Array(2))]

weights = np.zeros((3, 2)) + 3
bias = np.ones(2)

builder = NeuralNetworkBuilder(input_features, output_features)
builder.add_inner_product(name='ip_layer', W=weights, b=bias, input_channels=3, output_channels=2, has_bias=True, input_name='image', output_name='probs')

# compile the spec
mlmodel = ct.models.MLModel(builder.spec)

# trigger the ANE!
out = mlmodel.predict({"image": np.array([1337,0,0], dtype=np.float32)})
print(out)
mlmodel.save('test.mlmodel')

