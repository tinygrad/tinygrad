import ast, inspect, os, pathlib, shlex, subprocess, sys, textwrap, types, unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad import Tensor, dtypes
from tinygrad.llm import glm5next as flash
from tinygrad.llm import glm_dsa as full
from tinygrad.llm.kernels.amd_glm import FP8_ROW, Q4_ROW
from tinygrad.llm.kernels.amd_glm_dsa import Q4_ROPE_ROW, FP8_ROPE_ROW
from tinygrad.llm.kernels import amd_glm


def mock_environment(values):
  def get(key, default=None):
    value = values.get(key, default)
    return value if default is None or isinstance(default, str) else type(default)(str(value))
  return get

def select_dtype(block, values, dim=512):
  mod = sys.modules[block.__module__]
  owner = SimpleNamespace(config=SimpleNamespace(kv_lora_rank=512, rope_dim=dim-512, max_context=16,
                                                   cache_rows=0, rope_theta=10000., yarn=False), _regions=None, indexer=None)
  calls = []
  def empty(*shape, **kwargs):
    calls.append((shape, kwargs))
    return SimpleNamespace(shape=shape, dtype=kwargs['dtype'])
  with patch.object(mod, 'getenv', side_effect=mock_environment(values)), \
       patch.object(mod, 'seq_sharded', return_value=True), \
       patch.object(Tensor, 'empty', side_effect=empty), \
       patch.object(mod, 'precompute_freqs_cis', return_value=None):
    types.MethodType(block._init_state, owner)(SimpleNamespace(shape=(1, 1), device=('NULL:HIP:gfx1100', 'NULL:HIP:gfx1100')))
  assert calls, 'cache initialization must choose a storage layout'
  return calls[0][0], calls[0][1]['dtype']

class TestCacheFormatProfiles(unittest.TestCase):
  def test_c5_q4_and_fp8_are_not_the_same_layout(self):
    shape, dt = select_dtype(flash.MLATransformerBlock, {'SEQ_KV4': 1})
    self.assertEqual((shape[-1], dt), (Q4_ROW, dtypes.u32))
    shape, dt = select_dtype(flash.MLATransformerBlock, {'SEQ_KV4': 0, 'SEQ_FP8': 1})
    self.assertEqual((shape[-1], dt), (FP8_ROW, dtypes.u32))

  def test_explicit_fp8_off_retains_half(self):
    shape, dt = select_dtype(flash.MLATransformerBlock, {'SEQ_FP8': 0, 'SEQ_KV4': 0})
    self.assertEqual((shape[-1], dt), (512, dtypes.f16))

  def test_c9_default_is_half_and_q4_is_opt_in(self):
    shape, dt = select_dtype(full.MLATransformerBlock, {}, dim=576)
    self.assertEqual((shape[-1], dt), (576, dtypes.f16))
    for env in ({'GLM_KV': 'q4', 'SEQ_FP8': 1, 'SEQ_FP8_IDX': 0, 'SEQ_KV4': 1}, {'SEQ_FP8': 1, 'SEQ_FP8_IDX': 0, 'SEQ_KV4': 1}):
      with self.subTest(env=env):
        shape, dt = select_dtype(full.MLATransformerBlock, env, dim=576)
        self.assertEqual((shape[-1], dt), (Q4_ROPE_ROW, dtypes.u32))

  def test_c9_explicit_fp8_remains_available(self):
    shape, dt = select_dtype(full.MLATransformerBlock, {'GLM_KV': 'fp8', 'SEQ_FP8': 1, 'SEQ_FP8_IDX': 0}, dim=576)
    self.assertEqual((shape[-1], dt), (FP8_ROPE_ROW, dtypes.u32))


class TestC9LauncherProfiles(unittest.TestCase):
  ROOT = pathlib.Path(__file__).resolve().parents[2]

  def command(self, profile):
    env = {'PATH': os.environ['PATH'], 'HOME': os.environ['HOME'], 'VISION_MMPROJ': '/test-only/mmproj-F16.gguf', **profile}
    p = subprocess.run([self.ROOT / 'extra/llm_service/glm_14xr9700.sh', '--print-command'], cwd=self.ROOT, capture_output=True, text=True, env=env)
    self.assertEqual(p.returncode, 0, p.stderr)
    self.assertIn('PENDING: preview only', p.stderr)
    tokens = shlex.split(p.stdout)
    assert '--max_context' in tokens
    self.assertEqual(tokens[tokens.index('--max_context') + 1], profile.get('MAX_CONTEXT', '262144'))
    return {k: v for t in tokens if '=' in t and not t.startswith('--') for k, v in [t.split('=', 1)]}

  def test_q4_alias_maps_complete_profile(self):
    profile = {'GLM_KV': 'q4', 'MAX_CONTEXT': '1048576', 'SEQ_ROWS': '1048576', 'SEQ_SLOTS': '16',
               'SEQ_MAX_ACTIVE': '16', 'LOAD_DEFER': '1', 'P2P_GLM5_OFF': '1', 'FREQS_HOST': '1', 'AMD_SCRATCH_MAX': '64', 'SEQ_COPY_SCRATCH': '1'}
    for variance in (profile, {**profile, 'SEQ_FP8': '1', 'SEQ_FP8_IDX': '0', 'SEQ_KV4': '1'}):
      with self.subTest(explicit=all(k in variance for k in ('SEQ_FP8', 'SEQ_FP8_IDX', 'SEQ_KV4'))):
        env = self.command(variance)
        self.assertEqual((env['SEQ_FP8'], env['SEQ_FP8_IDX'], env['SEQ_KV4']), ('1', '0', '1'))
        for key, value in (('LOAD_DEFER', '1'), ('P2P_GLM5_OFF', '1'), ('FREQS_HOST', '1'), ('AMD_SCRATCH_MAX', '64'), ('SEQ_COPY_SCRATCH', '1')):
          self.assertEqual(env[key], value)
        shape, dt = select_dtype(full.MLATransformerBlock, env, dim=576)
        self.assertEqual((shape[-1], dt), (Q4_ROPE_ROW, dtypes.u32))

  def test_default_half(self):
    env = self.command({})
    for name in ('GLM_KV', 'SEQ_FP8', 'SEQ_FP8_IDX', 'SEQ_KV4', 'P2P_GLM5_OFF', 'FREQS_HOST'):
      self.assertNotIn(name, env)
    shape, dt = select_dtype(full.MLATransformerBlock, env, dim=576)
    self.assertEqual((shape[-1], dt), (576, dtypes.f16))

  def test_explicit_fp8_profile(self):
    env = self.command({'GLM_KV': 'fp8'})
    self.assertEqual((env['SEQ_FP8'], env['SEQ_FP8_IDX']), ('1', '0'))
    self.assertNotIn('SEQ_KV4', env)
    shape, dt = select_dtype(full.MLATransformerBlock, env, dim=576)
    self.assertEqual((shape[-1], dt), (FP8_ROPE_ROW, dtypes.u32))

  def test_invalid_profile_cannot_produce_a_command(self):
    env = dict(PATH=os.environ['PATH'], HOME=os.environ['HOME'], GLM_KV='q4', SEQ_FP8='0', VISION_MMPROJ='/test-only/mmproj-F16.gguf')
    result = subprocess.run([self.ROOT / 'extra/llm_service/glm_14xr9700.sh', '--print-command'],
                            cwd=self.ROOT, text=True, capture_output=True, env=env)
    self.assertEqual(result.returncode, 2)
    self.assertIn('requires SEQ_FP8=1', result.stderr)
    self.assertEqual(result.stdout, '')

def getenv_calls(tree, name):
  return [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
          node.func.id == 'getenv' and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == name]

class TestSchedulerProfileControls(unittest.TestCase):
  def test_idx_shard_lane_remains_environment_selectable(self):
    init = ast.parse(textwrap.dedent(inspect.getsource(flash.Indexer._init_state)))
    assert getenv_calls(init, 'SEQ_IDX_SHARD')
    full_init = ast.parse(textwrap.dedent(inspect.getsource(full.Indexer._init_state)))
    assert getenv_calls(full_init, 'SEQ_FP8_IDX')
    property_text = 'self.shard = len(cast(tuple, x.device)) if self.sparse and bool(getenv("SEQ_IDX_SHARD", 0)) and seq_sharded(x.device) else 0'
    assert property_text in inspect.getsource(flash.Indexer._init_state)
    for flag, expected in ((0, 0), (1, 2)):
      with self.subTest(SEQ_IDX_SHARD=flag):
        owner = SimpleNamespace(sparse=True)
        scope = {'self': owner, 'x': SimpleNamespace(device=('NULL', 'NULL:1')), 'sparse': owner.sparse,
                 'seq_sharded': lambda _: True, 'cast': lambda _t, val: val, 'getenv': mock_environment({'SEQ_IDX_SHARD': flag})}
        exec(property_text, scope)
        self.assertEqual(owner.shard, expected)
    tree = ast.parse(inspect.getsource(flash.Transformer))
    lanes = [node.test for node in ast.walk(tree) if isinstance(node, ast.IfExp) and isinstance(node.body, ast.Tuple) and
             any(isinstance(elt, ast.Name) and elt.id == 'IDX_LANE' for elt in node.body.elts)]
    self.assertEqual(len(lanes), 1)
    assert getenv_calls(lanes[0], 'SEQ_IDX_SHARD')

  def test_sample_top_k_respects_live_defaults_and_operator_choice(self):
    tree = ast.parse(inspect.getsource(flash))
    statements: list[ast.stmt] = [
      n for n in tree.body if isinstance(n, ast.Assign) and
      any(isinstance(t, ast.Name) and t.id == 'SAMPLE_TOP_K' for t in n.targets)
    ]
    self.assertEqual(len(statements), 1)
    def value(env):
      glb = {'getenv': lambda key, default=None: int(env[key]) if key in env else default}
      exec(compile(ast.Module(body=statements, type_ignores=[]), '<sample-top-k>', 'exec'), glb)
      return glb['SAMPLE_TOP_K']
    self.assertEqual(value({}), 0)
    self.assertEqual(value({'TOP_K': '20'}), 1)
    self.assertEqual(value({'SAMPLE_TOP_K': '1'}), 1)
    self.assertEqual(value({'TOP_K': '20', 'SAMPLE_TOP_K': '0'}), 0)

class TestMemoryLevers(unittest.TestCase):
  def test_freqs_host_enabled_only_with_tuple_and_flag(self):
    calls = []
    owner = SimpleNamespace(_freqs_host=lambda devs: calls.append(devs))
    statement = 'if getenv("FREQS_HOST", 0) and isinstance(devs, tuple): self._freqs_host(devs)'
    assert statement in inspect.getsource(full.Transformer._init_states)
    for enabled, expected in (('1', 1), (None, 0)):
      with self.subTest(enabled=enabled):
        calls.clear()
        scope = {'getenv': lambda key, default=None: enabled if key == 'FREQS_HOST' and enabled is not None else default,
                 'self': owner, 'devs': ('NULL', 'NULL:1')}
        exec(statement, scope)
        self.assertEqual(len(calls), expected)
        scope['devs'] = 'NULL'
        exec(statement, scope)
        self.assertEqual(len(calls), expected)

  def test_p2p_glm5_off_returns_before_allocation(self):
    amd_glm._P2P_OFF_SEEN.clear()
    try:
      with patch.object(amd_glm, '_p2p_allreduce', return_value='called') as wrapped, \
           patch.object(amd_glm, 'getenv', side_effect=lambda key, default=None: 1 if key == 'P2P_GLM5_OFF' else default), \
           redirect_stdout(StringIO()) as output:
        for _ in range(2): self.assertIsNone(amd_glm.p2p_allreduce(('NULL', 'NULL:1')))
      wrapped.assert_not_called()
      self.assertEqual(output.getvalue().count('P2P_GLM5_OFF:'), 1)
    finally:
      amd_glm._P2P_OFF_SEEN.clear()

if __name__ == '__main__': unittest.main()
