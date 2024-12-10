from __future__ import annotations
import ctypes, collections
from typing import List, Optional, Dict, Tuple, cast, Protocol, Type, Union, TypeVar, Generic, Any
# from tinygrad.runtime.autogen.am import am
from tinygrad.helpers import to_mv, round_up, getenv, mv_address

class TLSFAllocator:
  def __init__(self, size:int, base:int=0, block_size:int=16, lv2_cnt:int=16):
    self.size, self.base, self.block_size, self.l2_cnt = size, base, block_size, lv2_cnt.bit_length()
    self.storage = [collections.defaultdict(list) for _ in range(size.bit_length() + 1)]
    self.blocks = {0: (size, None, None, True)}  # size, next, prev (off size), is_free
    self._insert_block(0, size)

  def lv1(self, size): return size.bit_length()
  def lv2(self, size): return (size - (1 << (size.bit_length() - 1)))  // (1 << (size.bit_length() - self.l2_cnt))

  def _insert_block(self, start:int, size:int, prev:Optional[int]=None):
    if prev is None: prev = self.blocks[start][2]
    # print("insert", start, size, start + size, prev)
    self.storage[self.lv1(size)][self.lv2(size)].append(start)
    self.blocks[start] = (size, start + size, prev, True)
    return self

  def _remove_block(self, start:int, size:int, prev:Optional[int]=None):
    if prev is None: prev = self.blocks[start][2]
    # print("remove", start, size, start + size, prev)
    self.storage[self.lv1(size)][self.lv2(size)].remove(start)
    self.blocks[start] = (size, start + size, prev, False)
    return self

  def _verify_blocks(self):
    for start, (size, nxt, prev, is_free) in self.blocks.items():
      assert self.blocks.get(nxt) is None or self.blocks[nxt][2] == start, f"next block must point to current {nxt=}, {start=}"
      assert self.blocks.get(prev) is None or self.blocks[prev][1] == start, f"prev block must point to current, {prev=}, {start=}"
    return self

  def _split_block(self, start:int, size:int, new_size:int):
    nxt = self.blocks[start][1]
    assert self.blocks[start][3], "block must be free"
    self._remove_block(start, size)._insert_block(start, new_size)._insert_block(start + new_size, size - new_size, prev=start)
    if self.blocks.get(nxt) is not None: self.blocks[nxt] = (self.blocks[nxt][0], self.blocks[nxt][1], start + new_size, self.blocks[nxt][3])
    return self._verify_blocks()

  def _merge_right(self, start:int):
    size, nxt, prev, is_free = self.blocks[start]
    assert is_free, "block must be free"

    while is_free and self.blocks.get(nxt) is not None:
      if (blk:=self.blocks[nxt])[3] is False: break
      self._remove_block(start, size)._remove_block(nxt, blk[0])._insert_block(start, size:=size + blk[0])
      assert self.blocks[start][1] == blk[1]
      _, nxt, _, _ = self.blocks.pop(nxt)

    if self.blocks.get(nxt) is not None: self.blocks[nxt] = (self.blocks[nxt][0], self.blocks[nxt][1], start, self.blocks[nxt][3])
    self._verify_blocks()

  def _merge_block(self, start:int):
    while (x:=self.blocks[start][2]) is not None and self.blocks[x][3] is True: start = x
    self._merge_right(start)

  def alloc(self, o_size:int, align:int=1) -> int:
    o_size = max(self.block_size, o_size)
    size = max(self.block_size, o_size + align - 1)
    size = round_up(size, (1 << size.bit_length() - self.l2_cnt))

    for l1 in range(self.lv1(size), len(self.storage)):
      for l2 in range(self.lv2(size) if l1 == size.bit_length() else 0, (1 << self.l2_cnt)):
        if len(self.storage[l1][l2]) > 0:
          assert (nsize:=self.blocks[self.storage[l1][l2][0]][0]) >= size, "block must be larger"

          start = self.storage[l1][l2][0]
          if (new_start:=round_up(start, align)) != start:
            self._split_block(start, nsize, new_start - start)
            start, nsize = new_start, self.blocks[new_start][0]

          if nsize > o_size: self._split_block(start, nsize, o_size)
          self._remove_block(start, o_size)._verify_blocks()
          return start + self.base
    raise MemoryError("OOM")

  def free(self, start:int):
    self._insert_block(start - self.base, self.blocks[start - self.base][0])._merge_block(start - self.base)
