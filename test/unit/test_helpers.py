import ctypes, gzip, unittest, timeit, pickle, socket, urllib
from unittest.mock import patch
from tinygrad import Variable
from tinygrad.helpers import Context, ContextVar, argfix, colored, word_wrap, is_numpy_ndarray, mv_address, get_contraction, count, all_same
from tinygrad.helpers import merge_dicts, strip_parens, prod, round_up, fetch, fully_flatten, from_mv, to_mv, polyN, time_to_str, cdiv, cmod, getbits
from tinygrad.helpers import ceildiv
from tinygrad.tensor import Tensor, get_shape
import numpy as np

VARIABLE = ContextVar("VARIABLE", 0)

class TestContextVars(unittest.TestCase):
  # Ensuring that the test does not modify variables outside the tests.
  ctx = Context()
  def setUp(self): TestContextVars.ctx.__enter__()
  def tearDown(self): TestContextVars.ctx.__exit__()

  def test_initial_value_is_set(self):
    _TMP = ContextVar("_TMP", 5)
    self.assertEqual(_TMP.value, 5)

  def test_cannot_recreate(self):
    _TMP2 = ContextVar("_TMP2", 1)
    with self.assertRaises(RuntimeError):
      _TMP2 = ContextVar("_TMP2", 2)

  def test_new_var_inside_context(self):
    with Context(VARIABLE=1):
      _TMP3 = ContextVar("_TMP3", 1)
    with self.assertRaises(RuntimeError):
      _TMP3 = ContextVar("_TMP3", 2)

  def test_value_across_modules(self):
    # Mocking module import by invoking the code but not in our globals().
    exec('from tinygrad.helpers import ContextVar;C = ContextVar("C", 13)', {}) # pylint:disable=exec-used
    # It should not matter that the first creation was in another module.
    with self.assertRaises(RuntimeError):
      _C = ContextVar("C", 0)

  def test_assignment_across_modules(self):
    B = ContextVar("B", 1)
    # local assignment
    B.value = 2
    self.assertEqual(B.value, 2)
    with self.assertRaises(RuntimeError):
      # Assignment in another module.
      exec('from tinygrad.helpers import ContextVar;B = ContextVar("B", 0);B.value = 3;', {}) # pylint:disable=exec-used

  def test_context_assignment(self):
    with Context(VARIABLE=1):
      self.assertEqual(VARIABLE.value, 1)
    self.assertEqual(VARIABLE.value, 0)

  def test_unknown_param_to_context(self):
    with self.assertRaises(KeyError):
      with Context(SOMETHING_ELSE=1):
        pass

  def test_nested_context(self):
    with Context(VARIABLE=1):
      with Context(VARIABLE=2):
        MORE = ContextVar("MORE", 2)
        with Context(VARIABLE=3, MORE=3):
          self.assertEqual(VARIABLE.value, 3)
          self.assertEqual(MORE.value, 3)
        self.assertEqual(VARIABLE.value, 2)
        self.assertEqual(MORE.value, 2)
      self.assertEqual(VARIABLE.value, 1)
      self.assertEqual(MORE.value, 2)  # TODO: should this raise?
    self.assertEqual(VARIABLE.value, 0)

  def test_decorator(self):
    @Context(VARIABLE=1, DEBUG=4)
    def test():
      self.assertEqual(VARIABLE.value, 1)

    self.assertEqual(VARIABLE.value, 0)
    test()
    self.assertEqual(VARIABLE.value, 0)

  def test_context_exit_reverts_updated_values(self):
    D = ContextVar("D", 1)
    D.value = 2
    with Context(D=3):
      ...
    assert D.value == 2, f"Expected D to be 2, but was {D.value}. Indicates that Context.__exit__ did not restore to the correct value."

class TestAllSame(unittest.TestCase):
  def test_empty(self): self.assertTrue(all_same([]))
  def test_single(self): self.assertTrue(all_same([1]))
  def test_same(self): self.assertTrue(all_same([1, 1, 1]))
  def test_different(self): self.assertFalse(all_same([1, 2, 1]))

class TestMergeDicts(unittest.TestCase):
  def test_merge_dicts(self):
    a = {"a": 1, "b": 2}
    b = {"a": 1, "c": 3}
    c = {}
    d = {"a": 2, "b": 2}
    assert merge_dicts([a, b]) == {"a": 1, "b": 2, "c": 3}
    assert merge_dicts([a, c]) == a
    assert merge_dicts([a, b, c]) == {"a": 1, "b": 2, "c": 3}
    with self.assertRaises(RuntimeError):
      merge_dicts([a, d])

class TestStripParens(unittest.TestCase):
  def test_simple(self): self.assertEqual("1+2", strip_parens("(1+2)"))
  def test_nested(self): self.assertEqual("1+(2+3)", strip_parens("(1+(2+3))"))
  def test_casted_no_strip(self): self.assertEqual("(int)(1+2)", strip_parens("(int)(1+2)"))
  def test_unmatched_parens(self): self.assertEqual("((c35+c39>>23&255)+-127).cast(dtypes.float)",
    strip_parens("((c35+c39>>23&255)+-127).cast(dtypes.float)"))
  def test_single_paren_left(self): self.assertEqual("(abc", strip_parens("(abc"))
  def test_single_paren_right(self): self.assertEqual("abc)", strip_parens("abc)"))
  def test_parens_at_different_depths(self): self.assertEqual("(a+(b))*(c)", strip_parens("(a+(b))*(c)"))

class TestProd(unittest.TestCase):
  def test_empty(self): self.assertEqual(1, prod(tuple()))
  def test_ints(self): self.assertEqual(30, prod((2, 3, 5)))
  def test_variable(self): self.assertEqual("(a*12)", prod((Variable("a", 1, 5), 3, 4)).render())
  def test_variable_order(self): self.assertEqual("(a*12)", prod((3, 4, Variable("a", 1, 5))).render())

class TestRoundUp(unittest.TestCase):
  def test_round_up(self):
    self.assertEqual(round_up(-3,4), 0)
    self.assertEqual(round_up(-4,4), -4)
    self.assertEqual(round_up(6,4), 8)
    self.assertEqual(round_up(8,4), 8)
    self.assertEqual(round_up(232, 24984), 24984)
    self.assertEqual(round_up(24984, 232), 25056)

class TestCeilDiv(unittest.TestCase):
  def test_int(self):
    self.assertEqual(ceildiv(10, 3), 4)
    self.assertEqual(ceildiv(9, 3), 3)
    self.assertEqual(ceildiv(0, 5), 0)
    self.assertEqual(ceildiv(1, 5), 1)
  def test_symbolic(self):
    # tests that ceildiv with UOp uses (num + amt - 1) // amt formula for non-negative num
    v = Variable('v', 0, 100)
    result = ceildiv(v, 6)
    self.assertEqual(result.render(), "((v+5)//6)")
  def test_symbolic_negative_offset(self):
    # tests ceildiv(v-5, 6) which is used in conv2d output shape
    # old implementation incorrectly simplified -(x//-y) to ((v+1)//6-1) for v-5
    # new implementation uses (v-5+5)//6 = v//6 which is correct
    v = Variable('v', 11, 100)
    result = ceildiv(v - 5, 6)
    self.assertEqual(result.render(), "(v//6)")

class TestCount(unittest.TestCase):
  def test_count_basic(self):
    c = count(3)
    self.assertEqual(next(c), 3)
    self.assertEqual(next(c), 4)

  def test_count_step_pickle(self):
    c = count(1, 2)
    self.assertEqual(next(c), 1)
    c2 = pickle.loads(pickle.dumps(c))
    self.assertEqual(next(c2), 3)

@unittest.skip("no fetch tests because they need internet")
class TestFetch(unittest.TestCase):
  def test_fetch_bad_http(self):
    self.assertRaises(Exception, fetch, 'http://www.google.com/404', allow_caching=False)

  def test_fetch_small(self):
    assert (len(fetch('https://google.com', allow_caching=False).read_bytes())>0)

  def test_fetch_img(self):
    from PIL import Image
    img = fetch("https://avatars.githubusercontent.com/u/132956020", allow_caching=False)
    with Image.open(img) as pimg:
      assert pimg.size == (77, 77), pimg.size

  def test_fetch_subdir(self):
    from PIL import Image
    img = fetch("https://avatars.githubusercontent.com/u/132956020", allow_caching=False, subdir="images")
    with Image.open(img) as pimg:
      assert pimg.size == (77, 77), pimg.size
    assert img.parent.name == "images"

  def test_fetch_gunzip_valid(self):
    # compare fetch(gunzip=True) to fetch(gunzip=False) plus decompressing afterwards
    gzip_url: str = 'https://ftp.gnu.org/gnu/gzip/gzip-1.13.tar.gz'
    fp_gz = fetch(gzip_url, gunzip=True)
    fp_no_gz = fetch(gzip_url, gunzip=False)
    with open(fp_gz, 'rb') as f: content_gz = f.read()
    with open(fp_no_gz, 'rb') as f: content_no_gz = gzip.decompress(f.read())
    assert fp_gz.stat().st_size > fp_no_gz.stat().st_size
    assert isinstance(content_gz, bytes) and isinstance(content_no_gz, bytes)
    assert len(content_gz) == len(content_no_gz)
    assert content_gz == content_no_gz

  def test_fetch_gunzip_invalid(self):
    # given a non-gzipped file, fetch(gunzip=True) fails
    no_gzip_url: str = 'https://ftp.gnu.org/gnu/gzip/gzip-1.13.zip'
    with self.assertRaises(gzip.BadGzipFile):
      fetch(no_gzip_url, gunzip=True)

  def test_fetch_user_agent(self):
    fetch("https://csrc.nist.gov/CSRC/media/Projects/lightweight-cryptography/documents/finalist-round/updated-submissions/sparkle.zip",
          allow_caching=False)

  def test_fetch_half_and_full_file(self):
    x = fetch("https://csrc.nist.gov/CSRC/media/Projects/lightweight-cryptography/documents/finalist-round/updated-submissions/sparkle.zip",
          headers={"Range": "bytes=0-10"}).read_bytes()
    assert len(x) == 11, f"{len(x) != 11}"
    x = fetch("https://csrc.nist.gov/CSRC/media/Projects/lightweight-cryptography/documents/finalist-round/updated-submissions/sparkle.zip",
          headers={"Range": "bytes=0-100"}).read_bytes()
    assert len(x) == 101, f"{len(x) != 101}"

class TestFetchRetries(unittest.TestCase):

  def _test_retryable(self, error, retries=2):
    with patch('urllib.request.urlopen', side_effect=error) as mock_urlopen:
      with self.assertRaises(type(error)): fetch('http://example.com/test', allow_caching=False, retries=retries)
      assert mock_urlopen.call_count == retries + 1  # initial attempt + retries

  def _test_non_retryable(self, error):
    with patch('urllib.request.urlopen', side_effect=error) as mock_urlopen:
      with self.assertRaises(type(error)): fetch('http://example.com/test', allow_caching=False, retries=2)
      assert mock_urlopen.call_count == 1  # fails immediately, no retries

  # Retryable errors - should retry
  def test_fetch_connection_reset_error(self): self._test_retryable(ConnectionResetError())
  def test_fetch_socket_timeout(self): self._test_retryable(socket.timeout())
  def test_fetch_http_503(self): self._test_retryable(urllib.error.HTTPError(None, 503, None, None, None))
  def test_fetch_urlerror_timeout(self): self._test_retryable(urllib.error.URLError(TimeoutError()))

  # Non-retryable errors - should fail immediately
  def test_fetch_http_404_no_retry(self): self._test_non_retryable(urllib.error.HTTPError(None, 404, None, None, None))
  def test_fetch_value_error_no_retry(self): self._test_non_retryable(ValueError("invalid url"))
  def test_fetch_permission_error_no_retry(self): self._test_non_retryable(PermissionError())


class TestFullyFlatten(unittest.TestCase):
  def test_fully_flatten(self):
    self.assertEqual(fully_flatten([[1, 3], [1, 2]]), [1, 3, 1, 2])
    self.assertEqual(fully_flatten(((1, 3), (1, 2))), [1, 3, 1, 2])
    self.assertEqual(fully_flatten([[[1], [3]], [[1], [2]]]), [1, 3, 1, 2])
    self.assertEqual(fully_flatten([[[[1], 2], 3], 4]), [1, 2, 3, 4])
    self.assertEqual(fully_flatten([[1, 2, [3, 4]], [5, 6], 7]), [1, 2, 3, 4, 5, 6, 7])
    self.assertEqual(fully_flatten([[1, "ab"], [True, None], [3.14, [5, "b"]]]), [1, "ab", True, None, 3.14, 5, "b"])

  def test_fully_flatten_numpy(self):
    self.assertEqual(fully_flatten([np.array([])]), [])
    self.assertEqual(fully_flatten([np.array(3)]), [3])
    self.assertEqual(fully_flatten([np.array([3])]), [3])
    self.assertEqual(fully_flatten([np.array([[3]])]), [3])
    self.assertEqual(fully_flatten([np.array([1, 3]), np.array([1, 2])]), [1, 3, 1, 2])
    self.assertEqual(fully_flatten((np.array([1, 3]), np.array([1, 2]))), [1, 3, 1, 2])
    self.assertEqual(fully_flatten([np.array([[1], [3]]), np.array([[1], [2]])]), [1, 3, 1, 2])
    self.assertEqual(fully_flatten([[1, "ab"], [True, None], np.array([[3.14], [6.28]])]), [1, "ab", True, None, 3.14, 6.28])

class TestMemoryview(unittest.TestCase):
  def test_from_mv_to_mv(self):
    base = memoryview(bytearray(b"\x11\x22\x33"*40))
    ct = from_mv(base)
    mv = to_mv(ctypes.addressof(ct), len(base))
    mv[0] = 2
    assert base[0] == 2

  @unittest.skip("allocates tons of memory")
  def test_to_mv(self):
    sizes = [
      (16, "16 B"),
      (64, "64 B"),
      (256, "256 B"),
      (1024, "1 KB"),
      (4 * 1024, "4 KB"),
      (16 * 1024, "16 KB"),
      (64 * 1024, "64 KB"),
      (256 * 1024, "256 KB"),
      (1 * 1024 * 1024, "1 MB"),
      (10 * 1024 * 1024, "10 MB"),
      (200 * 1024 * 1024, "200 MB"),
    ]

    for sz, label in sizes:
      buf = np.random.randint(0, 256, sz, dtype=np.uint8)
      ptr = buf.ctypes.data

      iters = 100_000
      t_us = timeit.timeit(lambda: to_mv(ptr, sz), number=iters) * 1e6 / iters
      print(f"Size {label:>9} | Time: {t_us:8.3f} µs")

  def test_speed_from_mv_vs_mv_address(self):
    x = memoryview(bytearray(1))

    iters = 100000
    fmv_us = timeit.timeit(lambda: from_mv(x), number=iters) * 1e6 / iters
    mva_us = timeit.timeit(lambda: mv_address(x), number=iters) * 1e6 / iters
    print(f"from_mv vs mv_address: {fmv_us:8.3f} µs vs {mva_us:8.3f} µs")

class TestGetContraction(unittest.TestCase):
  def test_contraction(self):
    r = get_contraction((1,2,3,4), (2,3,4))
    self.assertEqual(r, [[0, 1], [2], [3]])

    r = get_contraction((2,1,3,4), (2,3,4))
    self.assertEqual(r, [[0], [1, 2], [3]])

    r = get_contraction((1,2,3,1,4), (1,2,3,4))
    self.assertEqual(r, [[], [0, 1], [2], [3, 4]])

    r = get_contraction((1,2,3,1,4,1,1), (2,3,4))
    self.assertEqual(r, [[0, 1], [2], [3, 4, 5, 6]])

    r = get_contraction((1,2,3,4), (1,2,3*4))
    self.assertEqual(r, [[], [0, 1], [2, 3]])

    r = get_contraction((1,2,3,4), (2,1,3,4))
    self.assertEqual(r, [[0, 1], [], [2], [3]])

    r = get_contraction((1,2,3,4), (1,1,2*3*4,1))
    self.assertEqual(r, [[], [], [0,1,2,3], []])

    r = get_contraction((2,1,3,4), (1,2,3,4))
    self.assertEqual(r, [[], [0], [1, 2], [3]])

    r = get_contraction((1,2,3,4), (2*3*4,1,1,1))
    self.assertEqual(r, [[0, 1, 2, 3], [], [], []])

    r = get_contraction((4,4,4,4), (16,1,16))
    self.assertEqual(r, [[0, 1], [], [2, 3]])

    r = get_contraction((1,2,3,4,1,1,1), (2,3,4))
    self.assertEqual(r, [[0, 1], [2], [3, 4, 5, 6]])

    r = get_contraction((1,2,3,4), (1,2,3,4,1))
    self.assertEqual(r, [[], [0, 1], [2], [3], []])

    r = get_contraction((14,1,384,14,1,1,1,1), (1,14,384,14))
    self.assertEqual(r, [[], [0], [1,2], [3,4,5,6,7]])

    r = get_contraction((14,1,384,1,14,1,1,1,1), (1,14,384,14))
    self.assertEqual(r, [[], [0], [1,2], [3,4,5,6,7,8]])

    r = get_contraction((512, 512), (1, 1, 512, 1, 1, 1, 1, 512))
    self.assertEqual(r, [[], [], [0], [], [], [], [], [1]])

    r = get_contraction((1,2,3,4), (1,2,6,2))
    self.assertEqual(r, None)

  def test_contraction_ones(self):
    r = get_contraction((1,), (1,1,1))
    self.assertEqual(r, [[], [], [0]])

    r = get_contraction((1,1), (1,1,1))
    self.assertEqual(r, [[], [], [0, 1]])

    r = get_contraction((1,1,1,1), (1,))
    self.assertEqual(r, [[0,1,2,3]])

    r = get_contraction((1,1,1,1), (1,1))
    self.assertEqual(r, [[], [0,1,2,3]])

    r = get_contraction((1,1,1,1), (1,1,1))
    self.assertEqual(r, [[], [], [0,1,2,3]])

    r = get_contraction((1,1,1,1), (1,1,1,1))
    self.assertEqual(r, [[], [], [], [0,1,2,3]])

class TestGetShape(unittest.TestCase):
  def test_get_shape(self):
    assert get_shape(2) == ()
    assert get_shape([]) == (0,)
    assert get_shape([[]]) == (1, 0)
    assert get_shape([[1, 2]]) == (1, 2)
    assert get_shape([[1, 2], (3, 4)]) == (2, 2)

  def test_inhomogeneous_shape(self):
    with self.assertRaises(ValueError): get_shape([[], [1]])
    with self.assertRaises(ValueError): get_shape([[1, [2]], [1]])
import unittest
import numpy as np
from tinygrad.helpers import polyN, is_numpy_ndarray
from tinygrad.tensor import Tensor

class TestPolyN(unittest.TestCase):
  def test_tensor(self):
    np.testing.assert_allclose(polyN(Tensor([1.0, 2.0, 3.0, 4.0]), [1.0, -2.0, 1.0]).numpy(), [0.0, 1.0, 4.0, 9.0])

class TestIsNumpyNdarray(unittest.TestCase):
  def test_tensor_numpy(self):
    self.assertTrue(is_numpy_ndarray(Tensor([1, 2, 3]).numpy()))

if __name__ == '__main__':
  unittest.main()
