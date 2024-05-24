::: tinygrad.Tensor
    options:
        heading_level: 2
        members: false
        show_source: false

## Properties

::: tinygrad.Tensor.shape
::: tinygrad.Tensor.dtype
::: tinygrad.Tensor.device

## Creation (basic)

::: tinygrad.Tensor.empty
::: tinygrad.Tensor.zeros
::: tinygrad.Tensor.ones
::: tinygrad.Tensor.full
::: tinygrad.Tensor.arange
::: tinygrad.Tensor.eye
::: tinygrad.Tensor.full_like
::: tinygrad.Tensor.zeros_like
::: tinygrad.Tensor.ones_like

## Creation (random)

::: tinygrad.Tensor.rand
::: tinygrad.Tensor.randn
::: tinygrad.Tensor.randint
::: tinygrad.Tensor.normal
::: tinygrad.Tensor.uniform
::: tinygrad.Tensor.scaled_uniform
::: tinygrad.Tensor.glorot_uniform
::: tinygrad.Tensor.kaiming_uniform
::: tinygrad.Tensor.kaiming_normal

## Data Access

::: tinygrad.Tensor.data
::: tinygrad.Tensor.item
::: tinygrad.Tensor.tolist
::: tinygrad.Tensor.numpy

## tinygrad ops

::: tinygrad.Tensor.schedule_with_vars
::: tinygrad.Tensor.schedule
::: tinygrad.Tensor.realize
::: tinygrad.Tensor.replace
::: tinygrad.Tensor.assign
::: tinygrad.Tensor.detach
::: tinygrad.Tensor.to
::: tinygrad.Tensor.to_
::: tinygrad.Tensor.shard
::: tinygrad.Tensor.shard_
::: tinygrad.Tensor.contiguous
::: tinygrad.Tensor.contiguous_backward
::: tinygrad.Tensor.backward

## Movement (low level)

::: tinygrad.Tensor.view
::: tinygrad.Tensor.reshape
::: tinygrad.Tensor.expand
::: tinygrad.Tensor.permute
::: tinygrad.Tensor.flip
::: tinygrad.Tensor.shrink
::: tinygrad.Tensor.pad

## Movement (high level)

::: tinygrad.Tensor.gather
::: tinygrad.Tensor.cat
::: tinygrad.Tensor.stack
::: tinygrad.Tensor.repeat
::: tinygrad.Tensor.split
::: tinygrad.Tensor.chunk
::: tinygrad.Tensor.squeeze
::: tinygrad.Tensor.unsqueeze
::: tinygrad.Tensor.pad2d
::: tinygrad.Tensor.T
::: tinygrad.Tensor.transpose
::: tinygrad.Tensor.flatten
::: tinygrad.Tensor.unflatten

## Reduce

::: tinygrad.Tensor.sum
::: tinygrad.Tensor.max
::: tinygrad.Tensor.min
::: tinygrad.Tensor.mean
::: tinygrad.Tensor.var
::: tinygrad.Tensor.std
::: tinygrad.Tensor.softmax
::: tinygrad.Tensor.log_softmax
::: tinygrad.Tensor.logsumexp
::: tinygrad.Tensor.argmax
::: tinygrad.Tensor.argmin

## Processing

::: tinygrad.Tensor.conv2d
::: tinygrad.Tensor.dot
::: tinygrad.Tensor.matmul
::: tinygrad.Tensor.einsum
::: tinygrad.Tensor.cumsum
::: tinygrad.Tensor.triu
::: tinygrad.Tensor.tril
::: tinygrad.Tensor.avg_pool2d
::: tinygrad.Tensor.max_pool2d
::: tinygrad.Tensor.conv_transpose2d

## Unary Ops (math)

::: tinygrad.Tensor.logical_not
::: tinygrad.Tensor.neg
::: tinygrad.Tensor.log
::: tinygrad.Tensor.log2
::: tinygrad.Tensor.exp
::: tinygrad.Tensor.exp2
::: tinygrad.Tensor.trunc
::: tinygrad.Tensor.ceil
::: tinygrad.Tensor.floor
::: tinygrad.Tensor.round
::: tinygrad.Tensor.lerp
::: tinygrad.Tensor.square
::: tinygrad.Tensor.clip
::: tinygrad.Tensor.abs
::: tinygrad.Tensor.sign
::: tinygrad.Tensor.reciprocal

## Unary Ops (activation)

::: tinygrad.Tensor.relu
::: tinygrad.Tensor.sigmoid
::: tinygrad.Tensor.elu
::: tinygrad.Tensor.celu
::: tinygrad.Tensor.swish
::: tinygrad.Tensor.silu
::: tinygrad.Tensor.relu6
::: tinygrad.Tensor.hardswish
::: tinygrad.Tensor.tanh
::: tinygrad.Tensor.sinh
::: tinygrad.Tensor.cosh
::: tinygrad.Tensor.atanh
::: tinygrad.Tensor.asinh
::: tinygrad.Tensor.acosh
::: tinygrad.Tensor.hardtanh
::: tinygrad.Tensor.gelu
::: tinygrad.Tensor.quick_gelu
::: tinygrad.Tensor.leakyrelu
::: tinygrad.Tensor.mish
::: tinygrad.Tensor.softplus
::: tinygrad.Tensor.softsign

## Elementwise Ops (broadcasted)

::: tinygrad.Tensor.add
::: tinygrad.Tensor.sub
::: tinygrad.Tensor.mul
::: tinygrad.Tensor.div
::: tinygrad.Tensor.xor
::: tinygrad.Tensor.lshift
::: tinygrad.Tensor.rshift
::: tinygrad.Tensor.pow
::: tinygrad.Tensor.maximum
::: tinygrad.Tensor.minimum
::: tinygrad.Tensor.where

## Neural Network Ops (functional)

::: tinygrad.Tensor.linear
::: tinygrad.Tensor.sequential
::: tinygrad.Tensor.layernorm
::: tinygrad.Tensor.batchnorm
::: tinygrad.Tensor.dropout
::: tinygrad.Tensor.one_hot
::: tinygrad.Tensor.scaled_dot_product_attention
::: tinygrad.Tensor.binary_crossentropy
::: tinygrad.Tensor.binary_crossentropy_logits
::: tinygrad.Tensor.sparse_categorical_crossentropy

## Casting Ops

::: tinygrad.Tensor.cast
::: tinygrad.Tensor.bitcast
::: tinygrad.Tensor.float
::: tinygrad.Tensor.half
