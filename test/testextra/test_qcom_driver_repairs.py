"""A630-3/4 contract regressions with inert host and ABI boundaries.

Addresses are keys into a bytearray. No native pointer, allocation, permission
probe, or host copy is used; the actual transaction and scheduler remain active.
"""
import collections, ctypes, unittest
from unittest.mock import patch
from test.mockgpu.qcom import qcomdriver as driver
from tinygrad.runtime.autogen import kgsl


class InertDriverTest(unittest.TestCase):
  def setUp(self):
    self.base = 4096
    self.storage = bytearray(range(32))
    self.transfers = []
    self.enterContext(patch.object(driver, "validate_host_mapping"))
    self.enterContext(patch.object(driver, "read_host", side_effect=self.read_host))
    self.enterContext(patch.object(driver, "write_host", side_effect=self.write_host))

  def read_host(self, address, size):
    self.assertTrue(self.base <= address <= address + size <= self.base + len(self.storage))
    self.transfers.append(("read", address, size))
    return bytes(self.storage[address - self.base:address - self.base + size])

  def write_host(self, address, data):
    self.assertTrue(self.base <= address <= address + len(data) <= self.base + len(self.storage))
    self.transfers.append(("write", address, len(data)))
    self.storage[address - self.base:address - self.base + len(data)] = data


class MappingIdentityRepairTests(InertDriverTest):
  def transaction(self, snapshot=False):
    space = driver.AddressSpace()
    first = space.map(self.base, len(self.storage), owner=self.storage)
    transaction = space.transaction()
    transaction.write(self.base, b"old")
    if snapshot: transaction.region(self.base, 4)
    return space, first, transaction

  def later_access(self, transaction, operation):
    if operation == "read": return transaction.read(self.base, 4)
    if operation == "validate": return transaction.validate(self.base, 4, write=True)
    if operation == "write": return transaction.write(self.base + 8, b"new")
    return transaction.region(self.base, 4)

  def test_replacement_rejects_every_access_before_cached_data_or_new_staging(self):
    # Replacing the retained record would mix old overlays/snapshots with a new
    # allocation. Equal field values still do not make two Mapping objects one.
    for snapshot in (False, True):
      for operation in ("read", "validate", "write", "region"):
        with self.subTest(snapshot=snapshot, operation=operation):
          space, first, transaction = self.transaction(snapshot)
          space.unmap(self.base)
          second = space.map(self.base, len(self.storage), owner=self.storage)
          self.assertEqual(first, second)
          overlays = list(transaction.overlays)
          snapshots = {base:bytes(data) for base, data in transaction.snapshots.items()}
          transfers = list(self.transfers)
          with self.assertRaisesRegex(ValueError, "mapping changed"):
            self.later_access(transaction, operation)
          self.assertIs(transaction.entries[self.base], first)
          self.assertIs(space.mappings[self.base], second)
          self.assertEqual(transaction.overlays, overlays)
          self.assertEqual(transaction.snapshots, snapshots)
          self.assertEqual(self.transfers, transfers)
          with self.assertRaisesRegex(ValueError, "mapping changed"): transaction.commit()
          self.assertEqual(bytes(self.storage), bytes(range(32)))
          self.assertFalse(transaction.finished)

  def test_immediate_commit_rejects_replacement_without_modifying_live_record(self):
    for snapshot in (False, True):
      with self.subTest(snapshot=snapshot):
        space, first, transaction = self.transaction(snapshot)
        space.unmap(self.base)
        second = space.map(self.base, len(self.storage), owner=self.storage)
        with self.assertRaisesRegex(ValueError, "mapping changed"): transaction.commit()
        self.assertIs(transaction.entries[self.base], first)
        self.assertIs(space.mappings[self.base], second)
        self.assertEqual(bytes(self.storage), bytes(range(32)))

  def test_unmapped_allocation_never_exposes_cached_data(self):
    for operation in ("read", "validate", "write", "region"):
      with self.subTest(operation=operation):
        space, first, transaction = self.transaction(snapshot=True)
        space.unmap(self.base)
        with self.assertRaisesRegex(ValueError, "unmapped"):
          self.later_access(transaction, operation)
        with self.assertRaisesRegex(ValueError, "mapping changed"): transaction.commit()
        self.assertIs(transaction.entries[self.base], first)
        self.assertEqual(bytes(self.storage), bytes(range(32)))

  def test_same_mapping_preserves_live_reads_overlay_order_and_region_updates(self):
    space, first, transaction = self.transaction()
    self.storage[8] = ord("L")
    self.assertEqual(transaction.read(self.base + 8, 1), b"L")
    transaction.write(self.base + 1, b"XY")
    _, region = transaction.region(self.base, 4)
    self.assertEqual(region[:3], b"oXY")
    transaction.validate(self.base, 4, write=True)
    transaction.write(self.base + 2, b"Z")
    self.assertEqual(transaction.read(self.base, 3), b"oXZ")
    self.assertIs(transaction.entries[self.base], first)
    transaction.commit()
    self.assertEqual(self.storage[:3], b"oXZ")
    self.assertIs(space.mappings[self.base], first)

  def test_fresh_transaction_can_use_replacement_after_old_transaction_rejects(self):
    space, _, old = self.transaction()
    space.unmap(self.base)
    second = space.map(self.base, len(self.storage), owner=self.storage)
    with self.assertRaisesRegex(ValueError, "mapping changed"): old.read(self.base, 3)
    fresh = space.transaction()
    fresh.write(self.base, b"new")
    fresh.commit()
    self.assertEqual(self.storage[:3], b"new")
    self.assertIs(space.mappings[self.base], second)
    with self.assertRaisesRegex(ValueError, "mapping changed"): old.commit()
    self.assertEqual(self.storage[:3], b"new")


class ConsumedTimestampRepairTests(InertDriverTest):
  def query(self, qcom, kind, context_id=1):
    req = kgsl.struct_kgsl_cmdstream_readtimestamp_ctxtid(context_id=context_id, type=kind)
    # Drop only the ABI response. A query drains queues, so staged publications
    # must still reach the inert storage through the actual transaction commit.
    def write(address, data):
      if address != 0: self.write_host(address, data)
    with patch.object(driver, "read_struct", return_value=req), patch.object(driver, "write_host", side_effect=write):
      qcom.ioctl(qcom.contexts[context_id].owner_fd, driver.ioctl_number(kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID), 0)
    return req.timestamp

  def timestamps(self, qcom, context_id=1):
    return tuple(self.query(qcom, kind, context_id) for kind in (
      kgsl.KGSL_TIMESTAMP_QUEUED, kgsl.KGSL_TIMESTAMP_CONSUMED, kgsl.KGSL_TIMESTAMP_RETIRED))

  def observe_without_scheduling(self, qcom, context_id=1):
    # A nested ioctl observes state while the scheduler re-entry latch is held.
    with patch.object(qcom, "executing", True): return self.timestamps(qcom, context_id)

  def test_queued_commands_become_consumed_only_when_their_fifo_turn_starts(self):
    released, seen = set(), []
    def execute(words, memory, timestamp, cursor):
      seen.append(words[0])
      return words[0] in released, cursor
    qcom = driver.QCOMDriver(execute)
    qcom.contexts[1] = driver.Context(owner_fd=7, queued=2,
                                    pending=collections.deque([((1,), 1, 0), ((2,), 2, 0)]))
    self.assertEqual(self.observe_without_scheduling(qcom), (2, 0, 0))
    for _ in range(3): self.assertEqual(self.timestamps(qcom), (2, 1, 0))
    self.assertEqual(set(seen), {1})
    self.assertEqual(qcom.contexts[1].pending[0][2], 0)
    released.add(1)
    self.assertEqual(self.timestamps(qcom), (2, 2, 1))
    released.add(2)
    self.assertEqual(self.timestamps(qcom), (2, 2, 2))
    self.assertFalse(qcom.contexts[1].pending)

  def test_prefix_wait_reports_start_before_retirement_and_resumes_once(self):
    actions = []
    def execute(words, memory, timestamp, start):
      for cursor in range(start, 3):
        if cursor == 0: memory.write(self.base, b"P")
        elif cursor == 1:
          if memory.read(self.base + 1, 1) != b"S": return False, cursor
        else: memory.write(self.base + 2, b"E")
        actions.append(cursor)
      return True, 3
    qcom = driver.QCOMDriver(execute)
    qcom.memory.map(self.base, len(self.storage), owner=self.storage)
    qcom.contexts[1] = driver.Context(owner_fd=7, queued=1, pending=collections.deque([((0,), 1, 0)]))
    for _ in range(3): self.assertEqual(self.timestamps(qcom), (1, 1, 0))
    self.assertEqual(self.storage[:3], b"P\x01\x02")
    self.assertEqual(actions, [0])
    self.assertEqual(qcom.contexts[1].pending[0][2], 1)
    self.storage[1] = ord("S")
    for _ in range(3): self.assertEqual(self.timestamps(qcom), (1, 1, 1))
    self.assertEqual(self.storage[:3], b"PSE")
    self.assertEqual(actions, [0, 1, 2])

  def test_contexts_keep_independent_started_and_retired_progress(self):
    qcom = driver.QCOMDriver(lambda words, memory, timestamp, cursor: (words[0] == 2, cursor))
    qcom.contexts[1] = driver.Context(owner_fd=7, queued=1, pending=collections.deque([((1,), 1, 0)]))
    qcom.contexts[2] = driver.Context(owner_fd=8, queued=2, pending=collections.deque([((2,), 1, 0), ((2,), 2, 0)]))
    self.assertEqual(self.timestamps(qcom, 1), (1, 1, 0))
    self.assertEqual(self.timestamps(qcom, 2), (2, 2, 2))
    self.assertEqual(self.timestamps(qcom, 1), (1, 1, 0))

  def test_invalid_whole_command_plan_does_not_publish_a_start(self):
    # Type-zero PM4 framing rejects in the real planner, before any host access.
    from test.mockgpu.qcom.qcomgpu import PM4Fault
    qcom = driver.QCOMDriver()
    qcom.contexts[1] = driver.Context(owner_fd=7, queued=1, pending=collections.deque([((0,), 1, 0)]))
    with self.assertRaisesRegex(PM4Fault, "unsupported PM4 packet type"): qcom.drain()
    self.assertEqual(self.observe_without_scheduling(qcom), (1, 0, 0))
    self.assertEqual(qcom.contexts[1].pending[0][2], 0)
    self.assertFalse(qcom.executing)
    self.assertEqual(self.transfers, [])

  def test_rejected_retry_retains_the_last_successful_start_observation(self):
    def execute(words, memory, timestamp, cursor):
      if cursor == 0: return False, 1
      raise ValueError("rejected retry")
    qcom = driver.QCOMDriver(execute)
    qcom.contexts[1] = driver.Context(owner_fd=7, queued=1, pending=collections.deque([((0,), 1, 0)]))
    with self.assertRaisesRegex(ValueError, "rejected retry"): qcom.drain()
    self.assertEqual(self.observe_without_scheduling(qcom), (1, 1, 0))
    self.assertEqual(qcom.contexts[1].pending[0][2], 1)
    self.assertFalse(qcom.executing)

  def test_invalid_execution_progress_does_not_publish_a_start(self):
    for result in ((False, -1), (False, "1"), (1, 1)):
      with self.subTest(result=result):
        qcom = driver.QCOMDriver(lambda *args: result)
        qcom.contexts[1] = driver.Context(owner_fd=7, queued=1, pending=collections.deque([((0,), 1, 0)]))
        with self.assertRaisesRegex(ValueError, "invalid PM4 execution progress"): qcom.drain()
        self.assertEqual(self.observe_without_scheduling(qcom), (1, 0, 0))
        self.assertEqual(qcom.contexts[1].pending[0][2], 0)

  def test_failed_commit_does_not_publish_a_start_or_staged_bytes(self):
    def execute(words, memory, timestamp, cursor):
      memory.write(self.base, b"bad")
      qcom.memory.unmap(self.base)
      qcom.memory.map(self.base, len(self.storage), owner=self.storage)
      return False, 1
    qcom = driver.QCOMDriver(execute)
    qcom.memory.map(self.base, len(self.storage), owner=self.storage)
    qcom.contexts[1] = driver.Context(owner_fd=7, queued=1, pending=collections.deque([((0,), 1, 0)]))
    with self.assertRaisesRegex(ValueError, "mapping changed"): qcom.drain()
    self.assertEqual(self.observe_without_scheduling(qcom), (1, 0, 0))
    self.assertEqual(self.storage, bytes(range(32)))
    self.assertEqual(qcom.contexts[1].pending[0][2], 0)

  def test_last_timestamp_is_consumed_and_retired_without_wrapping(self):
    ready = False
    qcom = driver.QCOMDriver(lambda *args: (ready, 0))
    qcom.memory.map(self.base, len(self.storage), owner=self.storage)
    qcom.contexts[1] = driver.Context(owner_fd=7, queued=0xfffffffe)
    obj = kgsl.struct_kgsl_command_object(gpuaddr=self.base, size=4, flags=kgsl.KGSL_CMDLIST_IB)
    def submit():
      req = kgsl.struct_kgsl_gpu_command(context_id=1, cmdsize=ctypes.sizeof(obj), numcmds=1)
      with patch.object(driver, "read_struct", side_effect=[req, obj]), patch.object(driver, "write_host"):
        qcom.ioctl(7, driver.ioctl_number(kgsl.IOCTL_KGSL_GPU_COMMAND), 0)
      return req.timestamp
    self.assertEqual(submit(), 0xffffffff)
    self.assertEqual(self.timestamps(qcom), (0xffffffff, 0xffffffff, 0))
    with self.assertRaisesRegex(ValueError, "timestamp exhausted"): submit()
    self.assertEqual(self.timestamps(qcom), (0xffffffff, 0xffffffff, 0))
    self.assertEqual(len(qcom.contexts[1].pending), 1)
    ready = True
    self.assertEqual(self.timestamps(qcom), (0xffffffff, 0xffffffff, 0xffffffff))


if __name__ == "__main__": unittest.main()
