import os

# The full CPU suite runs these on every PR. Sample fewer high-level suites on the secondary backend jobs,
# but keep device tests, renderer regressions, tensor cores, dtype coverage and unlisted (including new) files.
if os.getenv('SKIP_BACKEND_REDUNDANT') == '1':
  collect_ignore = [f'test_{name}.py' for name in (
    'arange', 'const_folding', 'conv', 'edgecases', 'function', 'getitem_ops', 'gradient', 'hashing', 'helpers', 'indexing',
    'invalid_tensor', 'linalg', 'llama_kernels', 'llm_mla', 'llm_moe', 'llm_server', 'masked_tensor', 'nn', 'optim',
    'range_movement', 'rearrange_einops', 'schedule', 'schedule_cache', 'setitem_schedule', 'softmax_fusion', 'stunning',
    'symbolic_jit', 'symbolic_ops', 'symbolic_tensor', 'tar', 'tensor', 'tensor_data', 'tensor_io', 'to_numpy', 'winograd')]
