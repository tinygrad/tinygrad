import unittest
from tinygrad.helpers import diskcache_get, diskcache_put

def remote_get(q,k): q.put(diskcache_get("test", k))
def remote_put(k,v): diskcache_put("test", k, v)

class DiskCache(unittest.TestCase):
  def test_putget(self):
    diskcache_put("test", "hello", "world")
    self.assertEqual(diskcache_get("test", "hello"), "world")
    diskcache_put("test", "hello", "world2")
    self.assertEqual(diskcache_get("test", "hello"), "world2")

  def test_putcomplex(self):
    diskcache_put("test", "k", ("complex", 123, "object"))
    ret = diskcache_get("test", "k")
    self.assertEqual(ret, ("complex", 123, "object"))

  def test_getotherprocess(self):
    from multiprocessing import Process, Queue
    diskcache_put("test", "k", "getme")
    q = Queue()
    p = Process(target=remote_get, args=(q,"k"))
    p.start()
    p.join()
    self.assertEqual(q.get(), "getme")

  def test_putotherprocess(self):
    from multiprocessing import Process
    p = Process(target=remote_put, args=("k", "remote"))
    p.start()
    p.join()
    self.assertEqual(diskcache_get("test", "k"), "remote")

  def test_no_table(self):
    self.assertIsNone(diskcache_get("faketable", "k"))

  def test_non_str_key(self):
    diskcache_put("test", 4, 5)
    self.assertEqual(diskcache_get("test", 4), 5)
    self.assertEqual(diskcache_get("test", "4"), 5)

if __name__ == "__main__":
  unittest.main()
