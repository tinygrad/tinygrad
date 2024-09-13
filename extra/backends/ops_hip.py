from __future__ import annotations
import ctypes
import functools
import subprocess
import io
from typing import Tuple, TypeVar, List, Any, cast, Set, Optional
import tinygrad.runtime.autogen.hip as hip
from tinygrad.helpers import (
    DEBUG, getenv, init_c_var, from_mv, round_up, to_mv,
    colored, init_c_struct_t
)
from tinygrad.device import (
    Compiled, LRUAllocator, BufferOptions, Runner, Device,
    Buffer, MallocAllocator, update_stats, Compiler, CompilerOptions
)
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.runtime.support.hip_comgr import compile_hip
from tinygrad.renderer.rdna import uops_to_rdna
from tinygrad.runtime.graph.hip import HIPGraph

# Constants
CHUNK_SIZE = 256 * 1024 * 1024  # 256 MB
PAGE_SIZE = 0x1000  # 4 KB
HIP_MEM_ALLOC_UNCACHED = 3
HIP_HOST_ALLOC_DEFAULT = 0
HIP_HOST_ALLOC_PORTABLE = 2

T = TypeVar("T")

# Global variable to track the current HIP device
hip_current_device: Optional[int] = None


def hip_set_device(device_id: int) -> None:
    """
    Sets the current HIP device if it's not already set.
    """
    global hip_current_device
    if device_id != hip_current_device:
        status = hip.hipSetDevice(device_id)
        check(status)
        hip_current_device = device_id


def check(status: int) -> None:
    """
    Checks the HIP status and raises an error if it indicates failure.
    """
    if status != 0:
        error_str = ctypes.string_at(hip.hipGetErrorString(status)).decode()
        raise RuntimeError(f"HIP Error {status}: {error_str}")


class RDNACompiler(Compiler):
    """
    Compiler for RDNA architecture.
    """
    linearizer_opts = LinearizerOptions("HIP", has_tensor_cores=True)

    def __init__(self, arch: str) -> None:
        super().__init__(f"compile_rdna_{arch}")
        self.arch = arch

    def render(self, name: str, uops: Any) -> str:
        return uops_to_rdna(name, uops)

    def compile(self, src: str) -> bytes:
        compiled_lib = compile_hip(src, self.arch, enable_rdna=True)
        if DEBUG >= 6:
            asm = subprocess.check_output(
                ["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'],
                input=compiled_lib
            )
            asm_filtered = '\n'.join(
                line for line in asm.decode('utf-8').split("\n") if 's_code_end' not in line
            )
            print(asm_filtered)
        return compiled_lib


class HIPCompiler(Compiler):
    """
    General HIP compiler.
    """
    compiler_opts = CompilerOptions("HIP", has_tensor_cores=True, shared_max=65536)

    def __init__(self, arch: str) -> None:
        super().__init__(f"compile_hip_{arch}")
        self.arch = arch

    def render(self, name: str, uops: Any) -> str:
        return HIPRenderer(name, uops)

    def compile(self, src: str) -> bytes:
        return compile_hip(src, self.arch)


class HIPProgram:
    """
    Represents a compiled HIP program.
    """
    def __init__(self, device: int, name: str, lib: bytes) -> None:
        self.device = device
        self.name = name
        self.lib = lib

        if DEBUG >= 6:
            asm = subprocess.check_output(
                ["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'],
                input=lib
            )
            asm_filtered = '\n'.join(
                line for line in asm.decode('utf-8').split("\n") if 's_code_end' not in line
            )
            print(asm_filtered)

        hip_set_device(self.device)
        self.module = init_c_var(
            hip.hipModule_t(),
            lambda x: check(hip.hipModuleLoadData(ctypes.byref(x), lib))
        )
        self.prg = init_c_var(
            hip.hipFunction_t(),
            lambda x: check(hip.hipModuleGetFunction(ctypes.byref(x), self.module, name.encode("utf-8")))
        )

    def __del__(self) -> None:
        if hasattr(self, 'module'):
            check(hip.hipModuleUnload(self.module))

    def __call__(
        self,
        *args: Any,
        global_size: Tuple[int, int, int] = (1, 1, 1),
        local_size: Tuple[int, int, int] = (1, 1, 1),
        vals: Tuple[int, ...] = (),
        wait: bool = False
    ) -> Optional[float]:
        hip_set_device(self.device)

        if not hasattr(self, "vargs"):
            struct_fields = [
                (f'f{i}', hip.hipDeviceptr_t) for i in range(len(args))
            ] + [
                (f'v{i}', ctypes.c_int) for i in range(len(vals))
            ]
            self.c_args = init_c_struct_t(tuple(struct_fields))(*args, *vals)
            self.vargs = (ctypes.c_void_p * 5)(
                ctypes.c_void_p(1),
                ctypes.cast(ctypes.byref(self.c_args), ctypes.c_void_p),
                ctypes.c_void_p(2),
                ctypes.cast(ctypes.byref(ctypes.c_size_t(ctypes.sizeof(self.c_args))), ctypes.c_void_p),
                ctypes.c_void_p(3)
            )
        else:
            for i, arg in enumerate(args):
                setattr(self.c_args, f'f{i}', arg)
            for i, val in enumerate(vals):
                setattr(self.c_args, f'v{i}', val)

        if wait:
            ev_start = init_c_var(hip.hipEvent_t(), lambda x: check(hip.hipEventCreate(ctypes.byref(x), 0)))
            ev_end = init_c_var(hip.hipEvent_t(), lambda x: check(hip.hipEventCreate(ctypes.byref(x), 0)))
            check(hip.hipEventRecord(ev_start, None))

        check(
            hip.hipModuleLaunchKernel(
                self.prg,
                *global_size,
                *local_size,
                0,
                None,
                None,
                self.vargs
            )
        )

        if wait:
            check(hip.hipEventRecord(ev_end, None))
            check(hip.hipEventSynchronize(ev_end))
            elapsed_time = ctypes.c_float()
            check(hip.hipEventElapsedTime(ctypes.byref(elapsed_time), ev_start, ev_end))
            check(hip.hipEventDestroy(ev_start))
            check(hip.hipEventDestroy(ev_end))
            return elapsed_time.value * 1e-3  # Convert ms to seconds

        return None


class HIPAllocator(LRUAllocator):
    """
    Allocator for HIP device memory.
    """
    def __init__(self, device: HIPDevice) -> None:
        self.device = device
        self.track_cross_device: Set[HIPDevice] = set()
        super().__init__()

    def full_synchronize(self) -> None:
        """
        Synchronizes the device and all tracked cross-device buffers.
        """
        self.device.synchronize()
        for dev in self.track_cross_device:
            dev.synchronize()
        self.track_cross_device.clear()

    def free_cache(self) -> None:
        """
        Frees the cache after a full synchronization.
        """
        self.full_synchronize()
        super().free_cache()

    def _alloc(self, size: int) -> hip.hipDeviceptr_t:
        """
        Allocates device memory.
        """
        hip_set_device(self.device.device)
        return init_c_var(
            hip.hipDeviceptr_t(),
            lambda x: check(hip.hipMalloc(ctypes.byref(x), size))
        )

    def _alloc_with_options(self, size: int, options: BufferOptions) -> hip.hipDeviceptr_t:
        """
        Allocates device memory with specific buffer options.
        """
        hip_set_device(self.device.device)
        if options.uncached:
            return init_c_var(
                hip.hipDeviceptr_t(),
                lambda x: check(hip.hipExtMallocWithFlags(ctypes.byref(x), size, HIP_MEM_ALLOC_UNCACHED))
            )
        elif options.host:
            flags = HIP_HOST_ALLOC_PORTABLE if options.signal else HIP_HOST_ALLOC_DEFAULT
            return init_c_var(
                hip.hipDeviceptr_t(),
                lambda x: check(hip.hipHostMalloc(ctypes.byref(x), size, flags))
            )
        else:
            raise ValueError("Unsupported BufferOptions")

    def _free(self, ptr: hip.hipDeviceptr_t) -> None:
        """
        Frees device memory.
        """
        check(hip.hipFree(ptr))

    def copy_from_fd(self, dest: hip.hipDeviceptr_t, fd: int, offset: int, size: int) -> None:
        """
        Copies data from a file descriptor to device memory.
        """
        hip_set_device(self.device.device)
        if not hasattr(self, 'hb'):
            self.hb = [self._alloc_with_options(CHUNK_SIZE, BufferOptions(host=True)) for _ in range(2)]
            self.hb_events: List[Optional[hip.hipEvent_t]] = [None, None]
            self.hb_polarity = 0

        with io.FileIO(fd, "rb", closefd=False) as fo:
            fo.seek(offset - (minor_offset := offset % PAGE_SIZE))
            copied_in = 0

            while copied_in < size:
                local_size = min(size - copied_in, CHUNK_SIZE)
                if self.hb_events[self.hb_polarity]:
                    check(hip.hipEventSynchronize(self.hb_events[self.hb_polarity]))
                    check(hip.hipEventDestroy(self.hb_events[self.hb_polarity]))
                    self.hb_events[self.hb_polarity] = None

                buffer = to_mv(self.hb[self.hb_polarity], local_size)
                bytes_read = fo.readinto(buffer)
                if bytes_read == 0:
                    break  # EOF reached

                copy_size = min(bytes_read - minor_offset, size - copied_in)
                check(hip.hipMemcpyAsync(
                    ctypes.c_void_p(dest.value + copied_in),
                    ctypes.c_void_p(self.hb[self.hb_polarity].value + minor_offset),
                    copy_size,
                    hip.hipMemcpyHostToDevice,
                    None
                ))

                self.hb_events[self.hb_polarity] = init_c_var(
                    hip.hipEvent_t(),
                    lambda x: check(hip.hipEventCreate(ctypes.byref(x)))
                )
                check(hip.hipEventRecord(self.hb_events[self.hb_polarity], None))

                copied_in += copy_size
                self.hb_polarity = (self.hb_polarity + 1) % len(self.hb)
                minor_offset = 0  # Only on the first iteration

    def copyin(self, dest: T, src: memoryview) -> None:
        """
        Copies data from host to device memory.
        """
        hip_set_device(self.device.device)
        host_mem = self._alloc_with_options(len(src), BufferOptions(host=True))
        self.device.pending_copyin.append(host_mem)
        ctypes.memmove(host_mem, from_mv(src), len(src))
        check(hip.hipMemcpyAsync(
            dest,
            host_mem,
            len(src),
            hip.hipMemcpyHostToDevice,
            None
        ))

    def copyout(self, dest: memoryview, src: T) -> None:
        """
        Copies data from device to host memory.
        """
        self.full_synchronize()
        hip_set_device(self.device.device)
        check(hip.hipMemcpy(
            from_mv(dest),
            src,
            len(dest),
            hip.hipMemcpyDeviceToHost
        ))

    def transfer(self, dest: T, src: T, size: int, **kwargs) -> None:
        """
        Transfers data between device memories.
        """
        hip_set_device(self.device.device)
        check(hip.hipMemcpyAsync(
            dest,
            src,
            size,
            hip.hipMemcpyDeviceToDevice,
            None
        ))


class HIPSyncEvent(Runner):
    """
    Runner for HIP synchronization events.
    """
    def __init__(self, lb: Any) -> None:
        self.lb = lb
        self.device: HIPDevice = cast(HIPDevice, Device[lb.device])
        self.dname = lb.device
        super().__init__()

    def __call__(
        self,
        rawbufs: List[Buffer],
        var_vals: Any,
        wait: bool = False,
        jit: bool = False
    ) -> None:
        to_mv(rawbufs[0]._buf, 4).cast("I")[0] = 0
        hip_set_device(self.device.device)
        check(hip.hipStreamWriteValue32(None, rawbufs[0]._buf, 1, 0))
        update_stats(
            colored("sync", "red"),
            0,
            0,
            {},
            None,
            1,
            jit,
            device=self.dname
        )


class HIPWaitEvent(Runner):
    """
    Runner for HIP wait events.
    """
    def __init__(self, device: str) -> None:
        self.device: HIPDevice = cast(HIPDevice, Device[device])
        self.dname = device
        super().__init__()

    def __call__(
        self,
        rawbufs: List[Buffer],
        var_vals: Any,
        wait: bool = False,
        jit: bool = False
    ) -> None:
        hip_set_device(self.device.device)
        check(hip.hipStreamWaitValue32(None, rawbufs[0]._buf, 1, 1, 0xFFFFFFFF))
        update_stats(
            colored("wait", "red"),
            0,
            0,
            {},
            None,
            1,
            jit,
            device=self.dname
        )


if getenv("HIPCPU"):
    rhip = ctypes.CDLL("/usr/local/lib/libremu.so")

    class RHIPProgram:
        """
        Represents a CPU-based HIP program for emulation.
        """
        def __init__(self, name: str, lib: bytes) -> None:
            self.name = name
            self.lib = lib

        def __call__(
            self,
            *args: Any,
            global_size: Tuple[int, int, int],
            local_size: Tuple[int, int, int],
            vals: Tuple[int, ...] = (),
            wait: bool = False
        ) -> None:
            all_args = (*args, *vals)
            arg_ptrs = (ctypes.c_void_p * len(all_args))(
                *[ctypes.cast(arg, ctypes.c_void_p) for arg in all_args]
            )
            rhip.hipModuleLaunchKernel(
                self.lib,
                len(self.lib),
                *global_size,
                *local_size,
                0,
                None,
                None,
                len(all_args),
                arg_ptrs
            )


class HIPDevice(Compiled):
    """
    Represents a HIP-compatible device.
    """
    def __init__(self, device: str = "") -> None:
        self.device = int(device.split(":")[1]) if ":" in device else 0
        self.pending_copyin: List[hip.hipDeviceptr_t] = []
        self.track_cross_buffer: List[Any] = []
        self.peers: Set[int] = set()

        if getenv("HIPCPU"):
            super().__init__(
                device,
                MallocAllocator,
                HIPCompiler("gfx1100"),
                RHIPProgram
            )
        else:
            device_props = init_c_var(
                hip.hipDeviceProp_t(),
                lambda x: check(hip.hipGetDeviceProperties(x, self.device))
            )
            self.arch = device_props.gcnArchName.decode()
            compiler = RDNACompiler(self.arch) if getenv("RDNA") else HIPCompiler(self.arch)
            super().__init__(
                device,
                HIPAllocator(self),
                compiler,
                functools.partial(HIPProgram, self.device),
                HIPGraph
            )

    def synchronize(self) -> None:
        """
        Synchronizes the device and cleans up pending operations.
        """
        if getenv("HIPCPU"):
            return
        hip_set_device(self.device)
        check(hip.hipDeviceSynchronize())
        for ptr in self.pending_copyin:
            check(hip.hipFree(ptr))
        self.track_cross_buffer.clear()
        self.pending_copyin.clear()

    def enable_peer(self, peer_device_num: int) -> None:
        """
        Enables peer access to another HIP device.
        """
        if self.device == peer_device_num or peer_device_num in self.peers:
            return
        hip_set_device(self.device)
        check(hip.hipDeviceEnablePeerAccess(peer_device_num, 0))
        self.peers.add(peer_device_num)
