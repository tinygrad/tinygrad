import ctypes, mmap, unittest

from tinygrad.helpers import DEV, IMAGE
from tinygrad.runtime.autogen import kgsl, libc, mesa
from tinygrad.runtime.ops_qcom import QCOMProgram, pkt4_hdr, pkt7_hdr, qreg
from test.mockgpu.qcom.qcomdriver import QCOMDriver
from test.mockgpu.qcom.qcomgpu import A630GPU


class TestA630CommandProcessor(unittest.TestCase):
  def test_program_rejects_oversized_global_dimensions(self):
    program = object.__new__(QCOMProgram)
    program.max_threads = 1024
    for axis in range(3):
      global_size = (65537 if axis == 0 else 1, 65537 if axis == 1 else 1, 65537 if axis == 2 else 1)
      with self.subTest(axis=axis), self.assertRaisesRegex(RuntimeError, "Invalid global/local dims"):
        program(global_size=global_size, local_size=(1, 1, 1))

  def test_register_packet(self):
    gpu = A630GPU(0)
    gpu.execute_words([pkt4_hdr(mesa.REG_A6XX_SP_CS_NDRANGE_0, 2), 0x1234, 0x5678])
    self.assertEqual(gpu.regs[mesa.REG_A6XX_SP_CS_NDRANGE_0], 0x1234)
    self.assertEqual(gpu.regs[mesa.REG_A6XX_SP_CS_NDRANGE_0 + 1], 0x5678)

  def test_signal_and_timestamp_packets(self):
    gpu = A630GPU(0)
    signal = ctypes.c_uint64()
    timestamp = ctypes.c_uint64()
    signal_addr, timestamp_addr = ctypes.addressof(signal), ctypes.addressof(timestamp)
    gpu.map_range(signal_addr, ctypes.sizeof(signal))
    gpu.map_range(timestamp_addr, ctypes.sizeof(timestamp))
    gpu.execute_words([
      pkt7_hdr(mesa.CP_EVENT_WRITE, 4), qreg.cp_event_write_0(event=mesa.CACHE_FLUSH_TS),
      signal_addr & 0xffffffff, signal_addr >> 32, 0x4567,
      pkt7_hdr(mesa.CP_REG_TO_MEM, 3),
      qreg.cp_reg_to_mem_0(reg=mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER, cnt=2, _64b=True),
      timestamp_addr & 0xffffffff, timestamp_addr >> 32,
    ])
    self.assertEqual(signal.value, 0x4567)
    self.assertGreater(timestamp.value, 0)

  def test_wait_rejects_unsatisfied_value(self):
    gpu = A630GPU(0)
    signal = ctypes.c_uint32(2)
    addr = ctypes.addressof(signal)
    gpu.map_range(addr, ctypes.sizeof(signal))
    packet = [pkt7_hdr(mesa.CP_WAIT_REG_MEM, 6),
              qreg.cp_wait_reg_mem_0(function=mesa.WRITE_GE, poll=mesa.POLL_MEMORY),
              addr & 0xffffffff, addr >> 32, 3, 0xffffffff, 32]
    with self.assertRaisesRegex(RuntimeError, "unsatisfied A630 memory wait"):
      gpu.execute_words(packet)


class TestQCOMDriverLifecycle(unittest.TestCase):
  def test_context_destroy_releases_context_state(self):
    driver = QCOMDriver()
    context = kgsl.struct_kgsl_drawctxt_create()
    driver.ioctl(0x13, ctypes.addressof(context))

    signal = ctypes.c_uint32()
    signal_address = ctypes.addressof(signal)
    words = (ctypes.c_uint32 * 7)(pkt7_hdr(mesa.CP_WAIT_REG_MEM, 6),
      qreg.cp_wait_reg_mem_0(function=mesa.WRITE_GE, poll=mesa.POLL_MEMORY),
      signal_address & 0xffffffff, signal_address >> 32, 1, 0xffffffff, 32)
    for obj in (signal, words):
      mapping = kgsl.struct_kgsl_map_user_mem(hostptr=ctypes.addressof(obj), len=ctypes.sizeof(obj))
      driver.ioctl(0x15, ctypes.addressof(mapping))
    command = kgsl.struct_kgsl_command_object(gpuaddr=ctypes.addressof(words), size=ctypes.sizeof(words))
    submission = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(command), numcmds=1, context_id=context.drawctxt_id,
                                               cmdsize=ctypes.sizeof(command))
    driver.ioctl(0x4a, ctypes.addressof(submission))
    self.assertEqual(len(driver.gpu.pending), 1)
    self.assertIn(context.drawctxt_id, driver.gpu.context_states)

    destroy = kgsl.struct_kgsl_drawctxt_destroy(drawctxt_id=context.drawctxt_id)
    driver.ioctl(0x14, ctypes.addressof(destroy))
    self.assertNotIn(context.drawctxt_id, driver.contexts)
    self.assertNotIn(context.drawctxt_id, driver.submitted_timestamps)
    self.assertNotIn(context.drawctxt_id, driver.gpu.context_states)
    self.assertNotIn(context.drawctxt_id, driver.gpu.completed_timestamps)
    self.assertEqual(driver.gpu.pending, [])
    with self.assertRaisesRegex(ValueError, "unknown KGSL context"):
      driver.ioctl(0x14, ctypes.addressof(destroy))

  def test_submission_timestamps_are_context_local(self):
    driver = QCOMDriver()
    contexts = [kgsl.struct_kgsl_drawctxt_create() for _ in range(2)]
    for context in contexts: driver.ioctl(0x13, ctypes.addressof(context))

    register = mesa.REG_A6XX_SP_CS_NDRANGE_0
    words = (ctypes.c_uint32 * 2)(pkt4_hdr(register, 1), 0x1234)
    mapping = kgsl.struct_kgsl_map_user_mem(hostptr=ctypes.addressof(words), len=ctypes.sizeof(words))
    driver.ioctl(0x15, ctypes.addressof(mapping))
    command = kgsl.struct_kgsl_command_object(gpuaddr=ctypes.addressof(words), size=ctypes.sizeof(words))

    timestamps = []
    for context in (contexts[0], contexts[1], contexts[0], contexts[1]):
      submission = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(command), numcmds=1, context_id=context.drawctxt_id,
                                                 cmdsize=ctypes.sizeof(command))
      driver.ioctl(0x4a, ctypes.addressof(submission))
      timestamps.append(submission.timestamp)

    context_ids = [context.drawctxt_id for context in contexts]
    self.assertEqual(timestamps, [1, 1, 2, 2])
    self.assertEqual(driver.submitted_timestamps, dict.fromkeys(context_ids, 2))
    self.assertEqual(driver.gpu.completed_timestamps, dict.fromkeys(context_ids, 2))
    self.assertEqual(driver.submitted_timestamp, 2)
    self.assertEqual(driver.completed_timestamp, 2)

  def test_external_mapping_rejects_wrapped_ranges(self):
    driver = QCOMDriver()
    mapping = kgsl.struct_kgsl_map_user_mem(hostptr=(1 << 64) - 0x100, len=0x1000)
    with self.assertRaisesRegex(ValueError, "overflows the 64-bit address space"):
      driver.ioctl(0x15, ctypes.addressof(mapping))
    self.assertEqual(mapping.gpuaddr, 0)
    self.assertEqual(driver.external_mappings, {})
    self.assertEqual(driver.gpu.mapped_ranges, {})
    self.assertEqual(driver.tracked_addresses, [])

  def test_gpuobj_allocation_rejects_oversized_requests(self):
    driver = QCOMDriver()
    for size in (0xfffff001, 0xffffffff, 1 << 32, (1 << 64) - 1):
      allocation = kgsl.struct_kgsl_gpuobj_alloc(size=size)
      with self.assertRaisesRegex(ValueError, "exceeds the A630 limit"):
        driver.ioctl(0x45, ctypes.addressof(allocation))
      self.assertEqual(allocation.id, 0)
      self.assertEqual(allocation.mmapsize, 0)
    self.assertEqual(driver.objects, {})
    self.assertEqual(driver.next_object_id, 1)

  def test_gpuobj_mmap_tracks_the_requested_extent(self):
    driver = QCOMDriver()
    allocation = kgsl.struct_kgsl_gpuobj_alloc(size=1, flags=kgsl.KGSL_MEMFLAGS_USE_CPU_MAP)
    driver.ioctl(0x45, ctypes.addressof(allocation))
    self.assertEqual(allocation.size, 0x1000)
    self.assertEqual(allocation.mmapsize, 0x1000)
    with self.assertRaisesRegex(ValueError, "not page aligned"):
      driver.mmap(0, allocation.mmapsize, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, allocation.id * 0x1000 + 1)
    with self.assertRaisesRegex(ValueError, "invalid KGSL mmap size"):
      driver.mmap(0, allocation.size + 0x1000, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, allocation.id * 0x1000)

    address = driver.mmap(0, allocation.mmapsize, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, allocation.id * 0x1000)
    try:
      self.assertEqual(driver.tracked_addresses[0][1] - driver.tracked_addresses[0][0], allocation.mmapsize)
      driver.ioctl(0x46, ctypes.addressof(kgsl.struct_kgsl_gpuobj_free(id=allocation.id)))
      self.assertEqual(driver.tracked_addresses, [])
    finally:
      libc.munmap(address, allocation.mmapsize)

    allocation = kgsl.struct_kgsl_gpuobj_alloc(size=1)
    driver.ioctl(0x45, ctypes.addressof(allocation))
    address = driver.mmap(0, allocation.size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, allocation.id * 0x1000)
    try:
      self.assertEqual(driver.tracked_addresses[0][1] - driver.tracked_addresses[0][0], allocation.size)
      driver.ioctl(0x46, ctypes.addressof(kgsl.struct_kgsl_gpuobj_free(id=allocation.id)))
      self.assertEqual(driver.tracked_addresses, [])
    finally:
      libc.munmap(address, allocation.size)

  def test_wait_in_one_context_does_not_block_another(self):
    driver = QCOMDriver()
    contexts = [kgsl.struct_kgsl_drawctxt_create() for _ in range(2)]
    for context in contexts: driver.ioctl(0x13, ctypes.addressof(context))

    signal = ctypes.c_uint32()
    signal_addr = ctypes.addressof(signal)
    register = mesa.REG_A6XX_SP_CS_NDRANGE_0
    wait_words = (ctypes.c_uint32 * 9)(pkt4_hdr(register, 1), 0x111,
      pkt7_hdr(mesa.CP_WAIT_REG_MEM, 6), qreg.cp_wait_reg_mem_0(function=mesa.WRITE_GE, poll=mesa.POLL_MEMORY),
      signal_addr & 0xffffffff, signal_addr >> 32, 7, 0xffffffff, 32)
    signal_words = (ctypes.c_uint32 * 7)(pkt4_hdr(register, 1), 0x222,
      pkt7_hdr(mesa.CP_EVENT_WRITE, 4), qreg.cp_event_write_0(event=mesa.CACHE_FLUSH_TS),
      signal_addr & 0xffffffff, signal_addr >> 32, 7)
    for obj in (signal, wait_words, signal_words):
      mapping = kgsl.struct_kgsl_map_user_mem(hostptr=ctypes.addressof(obj), len=ctypes.sizeof(obj))
      driver.ioctl(0x15, ctypes.addressof(mapping))

    for context, words in zip(contexts, (wait_words, signal_words)):
      command = kgsl.struct_kgsl_command_object(gpuaddr=ctypes.addressof(words), size=ctypes.sizeof(words))
      submission = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(command), numcmds=1, context_id=context.drawctxt_id,
                                                 cmdsize=ctypes.sizeof(command))
      driver.ioctl(0x4a, ctypes.addressof(submission))

    self.assertEqual(signal.value, 7)
    self.assertEqual(driver.gpu.pending, [])
    self.assertEqual(driver.gpu.context_states[contexts[0].drawctxt_id].regs[register], 0x111)
    self.assertEqual(driver.gpu.context_states[contexts[1].drawctxt_id].regs[register], 0x222)

  def test_rejected_submission_does_not_consume_timestamp(self):
    driver = QCOMDriver()
    context = kgsl.struct_kgsl_drawctxt_create()
    driver.ioctl(0x13, ctypes.addressof(context))
    submission = kgsl.struct_kgsl_gpu_command(context_id=context.drawctxt_id,
                                               cmdsize=ctypes.sizeof(kgsl.struct_kgsl_command_object))
    with self.assertRaisesRegex(ValueError, "no commands"):
      driver.ioctl(0x4a, ctypes.addressof(submission))
    self.assertEqual(driver.submitted_timestamp, 0)
    self.assertEqual(submission.timestamp, 0)

  def test_invalid_packet_does_not_poison_submission_queue(self):
    driver = QCOMDriver()
    context = kgsl.struct_kgsl_drawctxt_create()
    driver.ioctl(0x13, ctypes.addressof(context))
    invalid_words = (ctypes.c_uint32 * 4)(pkt7_hdr(mesa.CP_EXEC_CS, 3), 0, 1, 1)
    mapping = kgsl.struct_kgsl_map_user_mem(hostptr=ctypes.addressof(invalid_words), len=ctypes.sizeof(invalid_words))
    driver.ioctl(0x15, ctypes.addressof(mapping))
    command = kgsl.struct_kgsl_command_object(gpuaddr=ctypes.addressof(invalid_words), size=ctypes.sizeof(invalid_words))
    submission = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(command), numcmds=1, context_id=context.drawctxt_id,
                                               cmdsize=ctypes.sizeof(command))

    with self.assertRaisesRegex(ValueError, "CP_EXEC_CS expects 4 dwords"):
      driver.ioctl(0x4a, ctypes.addressof(submission))
    self.assertEqual(driver.submitted_timestamp, 0)
    self.assertEqual(submission.timestamp, 0)
    self.assertEqual(driver.gpu.pending, [])

    invalid_words = (ctypes.c_uint32 * 5)(pkt7_hdr(mesa.CP_EXEC_CS, 4), 1, 1, 1, 1)
    mapping = kgsl.struct_kgsl_map_user_mem(hostptr=ctypes.addressof(invalid_words), len=ctypes.sizeof(invalid_words))
    driver.ioctl(0x15, ctypes.addressof(mapping))
    command = kgsl.struct_kgsl_command_object(gpuaddr=ctypes.addressof(invalid_words), size=ctypes.sizeof(invalid_words))
    submission = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(command), numcmds=1, context_id=context.drawctxt_id,
                                               cmdsize=ctypes.sizeof(command))
    with self.assertRaisesRegex(ValueError, "CP_EXEC_CS expects a zero control dword"):
      driver.ioctl(0x4a, ctypes.addressof(submission))
    self.assertEqual(driver.submitted_timestamp, 0)
    self.assertEqual(submission.timestamp, 0)
    self.assertEqual(driver.gpu.pending, [])
    self.assertNotIn(context.drawctxt_id, driver.gpu.context_states)

    register = mesa.REG_A6XX_SP_CS_NDRANGE_0
    valid_words = (ctypes.c_uint32 * 2)(pkt4_hdr(register, 1), 0x1234)
    mapping = kgsl.struct_kgsl_map_user_mem(hostptr=ctypes.addressof(valid_words), len=ctypes.sizeof(valid_words))
    driver.ioctl(0x15, ctypes.addressof(mapping))
    command = kgsl.struct_kgsl_command_object(gpuaddr=ctypes.addressof(valid_words), size=ctypes.sizeof(valid_words))
    submission = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(command), numcmds=1, context_id=context.drawctxt_id,
                                               cmdsize=ctypes.sizeof(command))
    driver.ioctl(0x4a, ctypes.addressof(submission))
    self.assertEqual(submission.timestamp, 1)
    self.assertEqual(driver.gpu.context_states[context.drawctxt_id].regs[register], 0x1234)

  def test_execution_error_does_not_poison_submission_queue(self):
    driver = QCOMDriver()
    context = kgsl.struct_kgsl_drawctxt_create()
    driver.ioctl(0x13, ctypes.addressof(context))

    invalid_words = (ctypes.c_uint32 * 5)(pkt7_hdr(mesa.CP_EXEC_CS, 4), 0, 1, 1, 1)
    mapping = kgsl.struct_kgsl_map_user_mem(hostptr=ctypes.addressof(invalid_words), len=ctypes.sizeof(invalid_words))
    driver.ioctl(0x15, ctypes.addressof(mapping))
    command = kgsl.struct_kgsl_command_object(gpuaddr=ctypes.addressof(invalid_words), size=ctypes.sizeof(invalid_words))
    failed = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(command), numcmds=1, context_id=context.drawctxt_id,
                                           cmdsize=ctypes.sizeof(command))
    with self.assertRaisesRegex(RuntimeError, "kernel launch has no constant-buffer address"):
      driver.ioctl(0x4a, ctypes.addressof(failed))
    self.assertEqual(failed.timestamp, 1)
    self.assertEqual(driver.gpu.pending, [])

    register = mesa.REG_A6XX_SP_CS_NDRANGE_0
    valid_words = (ctypes.c_uint32 * 2)(pkt4_hdr(register, 1), 0x1234)
    mapping = kgsl.struct_kgsl_map_user_mem(hostptr=ctypes.addressof(valid_words), len=ctypes.sizeof(valid_words))
    driver.ioctl(0x15, ctypes.addressof(mapping))
    command = kgsl.struct_kgsl_command_object(gpuaddr=ctypes.addressof(valid_words), size=ctypes.sizeof(valid_words))
    recovered = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(command), numcmds=1, context_id=context.drawctxt_id,
                                              cmdsize=ctypes.sizeof(command))
    driver.ioctl(0x4a, ctypes.addressof(recovered))
    self.assertEqual(recovered.timestamp, 2)
    self.assertEqual(driver.gpu.completed_timestamps[context.drawctxt_id], 2)
    self.assertEqual(driver.gpu.context_states[context.drawctxt_id].regs[register], 0x1234)

  def test_unsupported_marker_does_not_consume_timestamp(self):
    driver = QCOMDriver()
    context = kgsl.struct_kgsl_drawctxt_create()
    driver.ioctl(0x13, ctypes.addressof(context))
    words = (ctypes.c_uint32 * 2)(pkt7_hdr(mesa.CP_SET_MARKER, 1),
      qreg.a6xx_cp_set_marker_0(mode=mesa.RM6_DIRECT_RENDER))
    mapping = kgsl.struct_kgsl_map_user_mem(hostptr=ctypes.addressof(words), len=ctypes.sizeof(words))
    driver.ioctl(0x15, ctypes.addressof(mapping))
    command = kgsl.struct_kgsl_command_object(gpuaddr=ctypes.addressof(words), size=ctypes.sizeof(words))
    submission = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(command), numcmds=1, context_id=context.drawctxt_id,
                                               cmdsize=ctypes.sizeof(command))

    with self.assertRaisesRegex(NotImplementedError, "unsupported A630 marker control"):
      driver.ioctl(0x4a, ctypes.addressof(submission))
    self.assertEqual(driver.submitted_timestamp, 0)
    self.assertEqual(submission.timestamp, 0)
    self.assertEqual(driver.gpu.pending, [])
    self.assertNotIn(context.drawctxt_id, driver.gpu.context_states)

  def test_bad_packet_parity_does_not_mutate_state(self):
    driver = QCOMDriver()
    context = kgsl.struct_kgsl_drawctxt_create()
    driver.ioctl(0x13, ctypes.addressof(context))
    register = mesa.REG_A6XX_SP_CS_NDRANGE_0
    packets = [
      ([pkt4_hdr(register, 1) ^ (1 << 7), 0x1234], "type-4"),
      ([pkt4_hdr(register, 1) ^ (1 << 27), 0x1234], "type-4"),
      ([pkt7_hdr(mesa.CP_WAIT_FOR_IDLE, 0) ^ (1 << 15)], "type-7"),
      ([pkt7_hdr(mesa.CP_WAIT_FOR_IDLE, 0) ^ (1 << 23)], "type-7"),
    ]
    for packet, packet_type in packets:
      with self.subTest(packet_type=packet_type, header=packet[0]):
        words = (ctypes.c_uint32 * len(packet))(*packet)
        mapping = kgsl.struct_kgsl_map_user_mem(hostptr=ctypes.addressof(words), len=ctypes.sizeof(words))
        driver.ioctl(0x15, ctypes.addressof(mapping))
        command = kgsl.struct_kgsl_command_object(gpuaddr=ctypes.addressof(words), size=ctypes.sizeof(words))
        submission = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(command), numcmds=1, context_id=context.drawctxt_id,
                                                   cmdsize=ctypes.sizeof(command))
        with self.assertRaisesRegex(ValueError, f"invalid A630 {packet_type} packet parity"):
          driver.ioctl(0x4a, ctypes.addressof(submission))
        self.assertEqual(driver.submitted_timestamp, 0)
        self.assertEqual(submission.timestamp, 0)
        self.assertEqual(driver.gpu.pending, [])
        self.assertNotIn(context.drawctxt_id, driver.gpu.context_states)

  def test_invalid_event_packets_do_not_consume_timestamp(self):
    driver = QCOMDriver()
    context = kgsl.struct_kgsl_drawctxt_create()
    driver.ioctl(0x13, ctypes.addressof(context))
    packets = [
      ([pkt7_hdr(mesa.CP_EVENT_WRITE, 1), qreg.cp_event_write_0(event=mesa.VS_DONE_TS)],
       NotImplementedError, "unsupported A630 event"),
      ([pkt7_hdr(mesa.CP_EVENT_WRITE, 2), qreg.cp_event_write_0(event=mesa.CACHE_INVALIDATE), 0xdeadbeef],
       ValueError, "CACHE_INVALIDATE expects 1 dwords"),
    ]
    for packet, error_type, error in packets:
      with self.subTest(error=error):
        words = (ctypes.c_uint32 * len(packet))(*packet)
        mapping = kgsl.struct_kgsl_map_user_mem(hostptr=ctypes.addressof(words), len=ctypes.sizeof(words))
        driver.ioctl(0x15, ctypes.addressof(mapping))
        command = kgsl.struct_kgsl_command_object(gpuaddr=ctypes.addressof(words), size=ctypes.sizeof(words))
        submission = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(command), numcmds=1, context_id=context.drawctxt_id,
                                                   cmdsize=ctypes.sizeof(command))
        with self.assertRaisesRegex(error_type, error): driver.ioctl(0x4a, ctypes.addressof(submission))
        self.assertEqual(driver.submitted_timestamp, 0)
        self.assertEqual(submission.timestamp, 0)
        self.assertEqual(driver.gpu.pending, [])
        self.assertNotIn(context.drawctxt_id, driver.gpu.context_states)

  def test_invalid_state_loads_do_not_consume_timestamp(self):
    driver = QCOMDriver()
    context = kgsl.struct_kgsl_drawctxt_create()
    driver.ioctl(0x13, ctypes.addressof(context))
    control = dict(state_type=mesa.ST_CONSTANTS, state_block=mesa.SB6_CS_SHADER, num_unit=1)
    packets = [
      ([pkt7_hdr(mesa.CP_LOAD_STATE6_FRAG, 3),
        qreg.cp_load_state6_0(**control, state_src=mesa.SS6_DIRECT), 0x111, 0x222],
       NotImplementedError, "unsupported A630 state source"),
      ([pkt7_hdr(mesa.CP_LOAD_STATE6_FRAG, 4),
        qreg.cp_load_state6_0(**control, state_src=mesa.SS6_INDIRECT), 0x1234, 0, 0xdeadbeef],
       ValueError, "CP_LOAD_STATE6_FRAG expects 3 dwords"),
      ([pkt7_hdr(mesa.CP_LOAD_STATE6_FRAG, 3),
        qreg.cp_load_state6_0(state_type=mesa.ST_SHADER, state_src=mesa.SS6_INDIRECT,
                              state_block=mesa.SB6_VS_SHADER, num_unit=1), 0x1234, 0],
       NotImplementedError, "unsupported A630 state load"),
    ]
    for packet, error_type, error in packets:
      with self.subTest(error=error):
        words = (ctypes.c_uint32 * len(packet))(*packet)
        mapping = kgsl.struct_kgsl_map_user_mem(hostptr=ctypes.addressof(words), len=ctypes.sizeof(words))
        driver.ioctl(0x15, ctypes.addressof(mapping))
        command = kgsl.struct_kgsl_command_object(gpuaddr=ctypes.addressof(words), size=ctypes.sizeof(words))
        submission = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(command), numcmds=1, context_id=context.drawctxt_id,
                                                   cmdsize=ctypes.sizeof(command))
        with self.assertRaisesRegex(error_type, error): driver.ioctl(0x4a, ctypes.addressof(submission))
        self.assertEqual(driver.submitted_timestamp, 0)
        self.assertEqual(submission.timestamp, 0)
        self.assertEqual(driver.gpu.pending, [])
        self.assertNotIn(context.drawctxt_id, driver.gpu.context_states)

  def test_unsupported_wait_semantics_do_not_consume_timestamp(self):
    driver = QCOMDriver()
    context = kgsl.struct_kgsl_drawctxt_create()
    driver.ioctl(0x13, ctypes.addressof(context))
    signal = ctypes.c_uint32(1)
    signal_addr = ctypes.addressof(signal)
    mapping = kgsl.struct_kgsl_map_user_mem(hostptr=signal_addr, len=ctypes.sizeof(signal))
    driver.ioctl(0x15, ctypes.addressof(mapping))

    invalid_controls = [
      (qreg.cp_wait_reg_mem_0(function=mesa.WRITE_NE, poll=mesa.POLL_MEMORY), "unsupported A630 wait function"),
      (qreg.cp_wait_reg_mem_0(function=mesa.WRITE_EQ, poll=mesa.POLL_REGISTER), "unsupported A630 wait poll mode"),
    ]
    for control, error in invalid_controls:
      with self.subTest(error=error):
        words = (ctypes.c_uint32 * 7)(pkt7_hdr(mesa.CP_WAIT_REG_MEM, 6), control,
          signal_addr & 0xffffffff, signal_addr >> 32, 1, 0xffffffff, 32)
        mapping = kgsl.struct_kgsl_map_user_mem(hostptr=ctypes.addressof(words), len=ctypes.sizeof(words))
        driver.ioctl(0x15, ctypes.addressof(mapping))
        command = kgsl.struct_kgsl_command_object(gpuaddr=ctypes.addressof(words), size=ctypes.sizeof(words))
        submission = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(command), numcmds=1, context_id=context.drawctxt_id,
                                                   cmdsize=ctypes.sizeof(command))
        with self.assertRaisesRegex(NotImplementedError, error): driver.ioctl(0x4a, ctypes.addressof(submission))
        self.assertEqual(driver.submitted_timestamp, 0)
        self.assertEqual(submission.timestamp, 0)
        self.assertEqual(driver.gpu.pending, [])

  def test_unsupported_reg_to_mem_does_not_return_timestamp(self):
    driver = QCOMDriver()
    context = kgsl.struct_kgsl_drawctxt_create()
    driver.ioctl(0x13, ctypes.addressof(context))
    destination = ctypes.c_uint64()
    destination_addr = ctypes.addressof(destination)
    register = mesa.REG_A6XX_SP_CS_NDRANGE_0
    words = (ctypes.c_uint32 * 6)(pkt4_hdr(register, 1), 0x1234,
      pkt7_hdr(mesa.CP_REG_TO_MEM, 3), qreg.cp_reg_to_mem_0(reg=register, cnt=2, _64b=True),
      destination_addr & 0xffffffff, destination_addr >> 32)
    for obj in (destination, words):
      mapping = kgsl.struct_kgsl_map_user_mem(hostptr=ctypes.addressof(obj), len=ctypes.sizeof(obj))
      driver.ioctl(0x15, ctypes.addressof(mapping))
    command = kgsl.struct_kgsl_command_object(gpuaddr=ctypes.addressof(words), size=ctypes.sizeof(words))
    submission = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(command), numcmds=1, context_id=context.drawctxt_id,
                                               cmdsize=ctypes.sizeof(command))

    with self.assertRaisesRegex(NotImplementedError, "unsupported A630 register-to-memory control"):
      driver.ioctl(0x4a, ctypes.addressof(submission))
    self.assertEqual(destination.value, 0)
    self.assertEqual(driver.submitted_timestamp, 0)
    self.assertEqual(submission.timestamp, 0)
    self.assertEqual(driver.gpu.pending, [])

  def test_null_command_list_is_rejected(self):
    driver = QCOMDriver()
    context = kgsl.struct_kgsl_drawctxt_create()
    driver.ioctl(0x13, ctypes.addressof(context))
    submission = kgsl.struct_kgsl_gpu_command(cmdlist=0, numcmds=1, context_id=context.drawctxt_id,
                                               cmdsize=ctypes.sizeof(kgsl.struct_kgsl_command_object))
    with self.assertRaisesRegex(ValueError, "command list pointer is null"):
      driver.ioctl(0x4a, ctypes.addressof(submission))
    self.assertEqual(driver.submitted_timestamp, 0)
    self.assertEqual(submission.timestamp, 0)
    self.assertEqual(driver.gpu.pending, [])

  def test_packets_cannot_span_command_objects(self):
    driver = QCOMDriver()
    context = kgsl.struct_kgsl_drawctxt_create()
    driver.ioctl(0x13, ctypes.addressof(context))
    register = mesa.REG_A6XX_SP_CS_NDRANGE_0
    command_words = [(ctypes.c_uint32 * 2)(pkt4_hdr(register, 2), 0x111), (ctypes.c_uint32 * 1)(0x222)]
    for words in command_words:
      mapping = kgsl.struct_kgsl_map_user_mem(hostptr=ctypes.addressof(words), len=ctypes.sizeof(words))
      driver.ioctl(0x15, ctypes.addressof(mapping))
    commands = (kgsl.struct_kgsl_command_object * 2)(*(kgsl.struct_kgsl_command_object(
      gpuaddr=ctypes.addressof(words), size=ctypes.sizeof(words)) for words in command_words))
    submission = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(commands), numcmds=2, context_id=context.drawctxt_id,
                                               cmdsize=ctypes.sizeof(kgsl.struct_kgsl_command_object))

    with self.assertRaisesRegex(ValueError, "truncated A630 type-4 packet"):
      driver.ioctl(0x4a, ctypes.addressof(submission))
    self.assertEqual(driver.submitted_timestamp, 0)
    self.assertEqual(submission.timestamp, 0)
    self.assertNotIn(context.drawctxt_id, driver.gpu.context_states)

  def test_untracks_freed_mappings(self):
    driver = QCOMDriver()
    allocation = kgsl.struct_kgsl_gpuobj_alloc(size=0x1000)
    driver.ioctl(0x45, ctypes.addressof(allocation))
    address = driver.mmap(0, allocation.mmapsize, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, allocation.id * 0x1000)
    try:
      self.assertEqual(len(driver.tracked_addresses), 1)
      free = kgsl.struct_kgsl_gpuobj_free(id=allocation.id)
      driver.ioctl(0x46, ctypes.addressof(free))
      self.assertEqual(driver.tracked_addresses, [])
    finally:
      libc.munmap(address, allocation.mmapsize)

    host = (ctypes.c_uint8 * 0x1000)()
    mapping = kgsl.struct_kgsl_map_user_mem(hostptr=ctypes.addressof(host), len=ctypes.sizeof(host))
    driver.ioctl(0x15, ctypes.addressof(mapping))
    self.assertEqual(len(driver.tracked_addresses), 1)
    shared_free = kgsl.struct_kgsl_sharedmem_free(gpuaddr=mapping.gpuaddr)
    driver.ioctl(0x21, ctypes.addressof(shared_free))
    self.assertEqual(driver.tracked_addresses, [])

  def test_duplicate_external_mapping_lifetime(self):
    driver = QCOMDriver()
    host = (ctypes.c_uint8 * 0x1000)()
    address = ctypes.addressof(host)
    for _ in range(2):
      mapping = kgsl.struct_kgsl_map_user_mem(hostptr=address, len=ctypes.sizeof(host))
      driver.ioctl(0x15, ctypes.addressof(mapping))
    self.assertEqual(len(driver.tracked_addresses), 2)

    shared_free = kgsl.struct_kgsl_sharedmem_free(gpuaddr=address)
    driver.ioctl(0x21, ctypes.addressof(shared_free))
    driver.gpu._validate_memory(address, ctypes.sizeof(host))
    self.assertEqual(len(driver.tracked_addresses), 1)

    driver.ioctl(0x21, ctypes.addressof(shared_free))
    with self.assertRaisesRegex(RuntimeError, "unmapped range"):
      driver.gpu._validate_memory(address, 1)
    self.assertEqual(driver.tracked_addresses, [])


@unittest.skipUnless(DEV.interface.startswith("MOCK") and DEV.device == "QCOM", "QCOM mock device required")
class TestQCOMEndToEnd(unittest.TestCase):
  def test_add(self):
    from tinygrad import Tensor
    self.assertEqual((Tensor([1, 2, 3]) + Tensor([4, 5, 6])).tolist(), [5, 7, 9])

  def test_reduce(self):
    from tinygrad import Tensor
    values = Tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
    self.assertEqual(values.sum(axis=1).tolist(), [6, 22, 38, 54])

  def test_scalar_and_where(self):
    from tinygrad import Tensor
    values = Tensor([-2, -1, 0, 1, 2])
    self.assertEqual((values > 0).where(values * 7 + 3, values * -1).tolist(), [2, 1, 0, 10, 17])

  def test_division(self):
    from tinygrad import Tensor
    self.assertEqual((Tensor([3.0, 8.0]) / Tensor([2.0, 4.0])).tolist(), [1.5, 2.0])

  def test_movement(self):
    from tinygrad import Tensor
    values = Tensor([[1, 2, 3], [4, 5, 6]])
    self.assertEqual(values.permute(1, 0).flip(0).contiguous().tolist(), [[3, 6], [2, 5], [1, 4]])

  def test_matmul(self):
    from tinygrad import Tensor
    left, right = Tensor([[1, 2, 3], [4, 5, 6]]), Tensor([[7, 8], [9, 10], [11, 12]])
    self.assertEqual((left @ right).tolist(), [[58, 64], [139, 154]])

  @unittest.skipUnless(IMAGE, "image lowering required")
  def test_image_matmul(self):
    from tinygrad import Tensor
    self.assertEqual((Tensor.ones(32, 32) @ Tensor.ones(32, 32)).tolist(), [[32.0] * 32 for _ in range(32)])


if __name__ == "__main__": unittest.main()
