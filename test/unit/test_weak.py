from tinygrad.helpers import LightWeakSet, LightWeakValueDictionary
import unittest
import time

CNT = 1000

cnt = 0
class MyObject:
  def __init__(self):
    global cnt
    self.cnt = cnt
    cnt += 1
    #print(f"object {self.cnt} created")
  #def __del__(self): print(f"object {self.cnt} destroyed")

class TestWeak(unittest.TestCase):
  def test_set_drops(self):
    ss = LightWeakSet()
    ss.add(MyObject())
    assert len(ss) == 0

  def test_set_holds(self):
    ss = LightWeakSet()
    obj = MyObject()
    ss.add(obj)
    assert len(ss) == 1

  def test_set_late_drops(self):
    ss = LightWeakSet()
    obj = MyObject()
    ss.add(obj)
    assert len(ss) == 1
    del obj
    assert len(ss) == 0

  def test_dict_drops(self):
    dd = LightWeakValueDictionary()
    dd[0] = MyObject()
    assert 0 not in dd

  def test_dict_holds(self):
    dd = LightWeakValueDictionary()
    dd[0] = ret = MyObject()
    assert 0 in dd

  def test_a_myobj_microbench(self):
    for _ in range(3):
      st = time.perf_counter_ns()
      for _ in range(CNT):
        obj = MyObject()
      et = (time.perf_counter_ns() - st)/CNT
      print(f"{et:.2f} ns to create MyObject")

  def test_set_add_microbench(self):
    for _ in range(3):
      ss = LightWeakSet()
      st = time.perf_counter_ns()
      for _ in range(CNT):
        obj = MyObject()
        ss.add(obj)
      assert len(ss) == 1
      et = (time.perf_counter_ns() - st)/CNT
      print(f"{et:.2f} ns to add to LightWeakSet")

  def test_set_del_microbench(self):
    for _ in range(3):
      ss = LightWeakSet()
      st = time.perf_counter_ns()
      for _ in range(CNT):
        obj = MyObject()
        ss.add(obj)
        ss.discard(obj)
      assert len(ss) == 0
      et = (time.perf_counter_ns() - st)/CNT
      print(f"{et:.2f} ns to add/del from LightWeakSet")

  def test_dict_add_microbench(self):
    for _ in range(3):
      dd = LightWeakValueDictionary()
      st = time.perf_counter_ns()
      for i in range(CNT):
        obj = MyObject()
        dd[i] = obj
      assert len(dd) == 1
      et = (time.perf_counter_ns() - st)/CNT
      print(f"{et:.2f} ns to add to LightWeakDict")

  def test_dict_check_microbench(self):
    for _ in range(3):
      dd = LightWeakValueDictionary()
      st = time.perf_counter_ns()
      for i in range(CNT):
        obj = MyObject()
        dd[i] = obj
        assert i in dd
        tst = dd[i]
        del obj,tst
      assert len(dd) == 0
      et = (time.perf_counter_ns() - st)/CNT
      print(f"{et:.2f} ns to add/del from LightWeakDict")

if __name__ == '__main__':
  unittest.main()