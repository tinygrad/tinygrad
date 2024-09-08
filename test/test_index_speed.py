from tinygrad import Tensor
import torch
import copy, time

def clone(original:Tensor): return copy.copy(original)

def get_item(tensor, indexer):
  tensor[indexer]

def set_item(tensor, indexer, val):
  pyt = clone(tensor)
  pyt[indexer] = val

reference = Tensor.arange(0., 160).reshape(4, 8, 5)
reference_t = torch.arange(0., 160).view(4, 8, 5)

indices_to_test = [
    [slice(None), slice(None), [0, 3, 4]],
    [slice(None), [2, 4, 5, 7], slice(None)],
    [[2, 3], slice(None), slice(None)],
    [slice(None), [0, 2, 3], [1, 3, 4]],
    [slice(None), [0], [1, 2, 4]],
    [slice(None), [0, 1, 3], [4]],
    [slice(None), [[0, 1], [1, 0]], [[2, 3]]],
    [slice(None), [[0, 1], [2, 3]], [[0]]],
    [slice(None), [[5, 6]], [[0, 3], [4, 4]]],
    [[0, 2, 3], [1, 3, 4], slice(None)],
    [[0], [1, 2, 4], slice(None)],
    [[0, 1, 3], [4], slice(None)],
    [[[0, 1], [1, 0]], [[2, 1], [3, 5]], slice(None)],
    [[[0, 1], [1, 0]], [[2, 3]], slice(None)],
    [[[0, 1], [2, 3]], [[0]], slice(None)],
    [[[2, 1]], [[0, 3], [4, 4]], slice(None)],
    [[[2]], [[0, 3], [4, 1]], slice(None)],
    # non-contiguous indexing subspace
    [[0, 2, 3], slice(None), [1, 3, 4]],

    # less dim, ellipsis
    [[0, 2], ],
    [[0, 2], slice(None)],
    [[0, 2], Ellipsis],
    [[0, 2], slice(None), Ellipsis],
    [[0, 2], Ellipsis, slice(None)],
    [[0, 2], [1, 3]],
    [[0, 2], [1, 3], Ellipsis],
    [Ellipsis, [1, 3], [2, 3]],
    [Ellipsis, [2, 3, 4]],
    [Ellipsis, slice(None), [2, 3, 4]],
    [slice(None), Ellipsis, [2, 3, 4]],

    # ellipsis counts for nothing
    [Ellipsis, slice(None), slice(None), [0, 3, 4]],
    [slice(None), Ellipsis, slice(None), [0, 3, 4]],
    [slice(None), slice(None), Ellipsis, [0, 3, 4]],
    [slice(None), slice(None), [0, 3, 4], Ellipsis],
    [Ellipsis, [[0, 1], [1, 0]], [[2, 1], [3, 5]], slice(None)],
    [[[0, 1], [1, 0]], [[2, 1], [3, 5]], Ellipsis, slice(None)],
    [[[0, 1], [1, 0]], [[2, 1], [3, 5]], slice(None), Ellipsis],
]
get_time, get_time_t, set_time, set_time_t = 0, 0, 0, 0

start = time.time()
for indexer in indices_to_test: get_item(reference, indexer)
end = time.time()
get_time += end - start

start = time.time()
for indexer in indices_to_test: get_item(reference_t, indexer)
end = time.time()
get_time_t += end - start

start = time.time()
for indexer in indices_to_test: set_item(reference, indexer, 212)
end = time.time()
set_time += end - start

start = time.time()
for indexer in indices_to_test: set_item(reference_t, indexer, 212)
end = time.time()
set_time_t += end - start

reference = Tensor.arange(0., 1296).reshape(3, 9, 8, 6)
reference_t = torch.arange(0., 1296).view(3, 9, 8, 6)

indices_to_test = [
    [slice(None), slice(None), slice(None), [0, 3, 4]],
    [slice(None), slice(None), [2, 4, 5, 7], slice(None)],
    [slice(None), [2, 3], slice(None), slice(None)],
    [[1, 2], slice(None), slice(None), slice(None)],
    [slice(None), slice(None), [0, 2, 3], [1, 3, 4]],
    [slice(None), slice(None), [0], [1, 2, 4]],
    [slice(None), slice(None), [0, 1, 3], [4]],
    [slice(None), slice(None), [[0, 1], [1, 0]], [[2, 3]]],
    [slice(None), slice(None), [[0, 1], [2, 3]], [[0]]],
    [slice(None), slice(None), [[5, 6]], [[0, 3], [4, 4]]],
    [slice(None), [0, 2, 3], [1, 3, 4], slice(None)],
    [slice(None), [0], [1, 2, 4], slice(None)],
    [slice(None), [0, 1, 3], [4], slice(None)],
    [slice(None), [[0, 1], [3, 4]], [[2, 3], [0, 1]], slice(None)],
    [slice(None), [[0, 1], [3, 4]], [[2, 3]], slice(None)],
    [slice(None), [[0, 1], [3, 2]], [[0]], slice(None)],
    [slice(None), [[2, 1]], [[0, 3], [6, 4]], slice(None)],
    [slice(None), [[2]], [[0, 3], [4, 2]], slice(None)],
    [[0, 1, 2], [1, 3, 4], slice(None), slice(None)],
    [[0], [1, 2, 4], slice(None), slice(None)],
    [[0, 1, 2], [4], slice(None), slice(None)],
    [[[0, 1], [0, 2]], [[2, 4], [1, 5]], slice(None), slice(None)],
    [[[0, 1], [1, 2]], [[2, 0]], slice(None), slice(None)],
    [[[2, 2]], [[0, 3], [4, 5]], slice(None), slice(None)],
    [[[2]], [[0, 3], [4, 5]], slice(None), slice(None)],
    [slice(None), [3, 4, 6], [0, 2, 3], [1, 3, 4]],
    [slice(None), [2, 3, 4], [1, 3, 4], [4]],
    [slice(None), [0, 1, 3], [4], [1, 3, 4]],
    [slice(None), [6], [0, 2, 3], [1, 3, 4]],
    [slice(None), [2, 3, 5], [3], [4]],
    [slice(None), [0], [4], [1, 3, 4]],
    [slice(None), [6], [0, 2, 3], [1]],
    [slice(None), [[0, 3], [3, 6]], [[0, 1], [1, 3]], [[5, 3], [1, 2]]],
    [[2, 2, 1], [0, 2, 3], [1, 3, 4], slice(None)],
    [[2, 0, 1], [1, 2, 3], [4], slice(None)],
    [[0, 1, 2], [4], [1, 3, 4], slice(None)],
    [[0], [0, 2, 3], [1, 3, 4], slice(None)],
    [[0, 2, 1], [3], [4], slice(None)],
    [[0], [4], [1, 3, 4], slice(None)],
    [[1], [0, 2, 3], [1], slice(None)],
    [[[1, 2], [1, 2]], [[0, 1], [2, 3]], [[2, 3], [3, 5]], slice(None)],

    # less dim, ellipsis
    [Ellipsis, [0, 3, 4]],
    [Ellipsis, slice(None), [0, 3, 4]],
    [Ellipsis, slice(None), slice(None), [0, 3, 4]],
    [slice(None), Ellipsis, [0, 3, 4]],
    [slice(None), slice(None), Ellipsis, [0, 3, 4]],
    [slice(None), [0, 2, 3], [1, 3, 4]],
    [slice(None), [0, 2, 3], [1, 3, 4], Ellipsis],
    [Ellipsis, [0, 2, 3], [1, 3, 4], slice(None)],
    [[0], [1, 2, 4]],
    [[0], [1, 2, 4], slice(None)],
    [[0], [1, 2, 4], Ellipsis],
    [[0], [1, 2, 4], Ellipsis, slice(None)],
    [[1], ],
    [[0, 2, 1], [3], [4]],
    [[0, 2, 1], [3], [4], slice(None)],
    [[0, 2, 1], [3], [4], Ellipsis],
    [Ellipsis, [0, 2, 1], [3], [4]],
    [slice(None), slice(None), [[0, 1], [1, 0]], [[2, 3], [3, 0]]],
    [slice(None), slice(None), [[2]], [[0, 3], [4, 4]]],
]

start = time.time()
for indexer in indices_to_test: get_item(reference, indexer)
end = time.time()
get_time += end - start

start = time.time()
for indexer in indices_to_test: get_item(reference_t, indexer)
end = time.time()
get_time_t += end - start

start = time.time()
for indexer in indices_to_test: set_item(reference, indexer, 1333)
end = time.time()
set_time += end - start

start = time.time()
for indexer in indices_to_test: set_item(reference_t, indexer, 1333)
end = time.time()
set_time_t += end - start

print(f"tinygrad get: {get_time}")
print(f"torch get: {get_time_t}")
print(f"tinygrad set: {set_time}")
print(f"torch set: {set_time_t}")
