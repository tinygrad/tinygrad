# MT generated launch and HCQ ABI

This backend targets the MTT S80 (QUYUAN1, BVNC `35.4.1632.23`). The launch path is generated in Python from MUSA ELF metadata. It does not use captured launch templates, a MUSA runtime launch generator, or MUSA userspace libraries.

## References

- MUSA 3.3.0 FWIF headers (`mtfw_fwif_*.h`) for CCB items, submission regions, responses, and firmware layout.
- `rogue_cdm.xml` (`CDMCTRL_KERNEL0` through `CDMCTRL_KERNEL11`) for the compute-grid register layout.
- Linux Imagination `pvr_mmu.c` for MMU4 page tables.
- Mesa `pvr_pds.c` for PDS instruction encoding and workgroup/local ID setup.

## Address map

`GENERATED_OFFSETS` in `generated.py` is an allocation map from device virtual addresses to backing-store offsets. It is fixed by the generation-1 launch ABI, not a catalog of kernel-specific launch data. Generated regions are mapped in full; `_write_backing()` takes a **backing-store offset**, not a device VA.

| Name | Purpose |
| --- | --- |
| `SUBMISSION_VA` | CDM submission packet and linked command stream. |
| `ARGS_VA` | Implicit and user argument page. Global spill base is at `+0`, physical core count at `+8`, and user arguments start at `+0x38`. |
| `GRID_VA` | CDM `KERNEL0`–`KERNEL8` state. Global sizes are at `+0x2c`, `+0x30`, and `+0x34`; local dimensions are packed at `+0x3c`. |
| `RESOURCE_VA` | Launch resource descriptor: argument-page pointer, shared-register count, control words, text offset, and temporary-register byte count. |
| `TEXT_VA` | USC instruction text. Spill text uses the 32 KiB region when it fits and the 192 KiB region for larger images. Other text follows the 4/8/192 KiB size classes. |
| `RESUME_VA` | Static resume PDS program (16 KiB). |
| `PDS_MAIN_VA` | Static main PDS program, followed by the kernel-specific PDS overlay. |
| `TASK_VA` | CDM task/context page. |
| `CDM_EXECUTION_ADDRESS` | Error/status region and PDS execution address. |

`ARGS_VA`, `GRID_VA`, `RESOURCE_VA`, `TEXT_VA`, `RESUME_VA`, `PDS_MAIN_VA`, and `TASK_VA` are defined in `submission.py` and shared by generation, queue construction, and tests.

## Launch generation

1. `MTCompiler.compile_raw()` invokes `mcc`, `llvm-objcopy`, and `clang-offload-bundler` to produce an MTGPU ELF.
2. `MTKernelImage.from_elf()` extracts `.text` and the MessagePack note. Kernel arguments, IDs, barriers, text entry, register counts, private memory, and immediates come from that metadata.
3. `MTGeneratedLaunch` generates the task page, resource descriptor, static main/resume PDS, and kernel-specific PDS for local IDs, workgroup IDs, and barriers. Text and all generated regions are mapped in full.
4. `activate_launch()` places text, sets the grid and implicit arguments, and allocates/maps private spill storage if required.
5. `MTComputeQueue.exec()` records an `MTSubmission` containing the launch, runtime-patched argument page, packet, sparse `write_map`, MMU generation, optional MMU snapshot, and optional symbolic-grid patch.
6. `submit()` uploads only changed backing pages and publishes the packet in the actual CCB request-ring cell. The queue index wraps at 256; response tokens use an independent absolute sequence.
7. Response value and token are validated before completion. Errors include fault-register diagnostics rather than being treated as successful execution.

## Spill/private memory

The spill arena is device memory, not host SVM memory. Its size is computed from `.private_memory_size` or `.local_mem_used_in_cluster`:

- Threads are `global_size * local_size`; per-core threads are `ceil(threads / physical_cores)`.
- The per-core stride is a power of two, with a 32 MiB minimum, covering `spill_bytes * per_core_threads`.
- The complete `stride * physical_cores` range is mapped with writable non-Meta MMU4 PTEs. Allocations above 1 GiB are rejected before submission.
- `Global Spill Base` is the arena VA and `Core Number` is the physical core count.
- Spill text uses the 32 KiB region when it fits and the 192 KiB region otherwise.
- `MT_SPILL_WORKGROUP_KERNEL8` (`0x108085`) is used for spill launches with more than one local thread.

These rules come from the MUSA launch ABI and are covered by `test/null/test_mtlaunch.py` and the conv3d tests in `test/runtime/test_ops.py`.

## HCQ2 integration

MT uses the same generic HCQ2 `pm_lower_call` path as AMD and NV: command bodies are encoded by `Compiled.pm_encode`, dependencies are lifted by the generic lifter, and links are resolved by the shared linker. There is no MT-specific direct-lowering function or alternate queue framework.

`serialize_hcq` is required because launches share one MMU root, generated backing store, and CCB item storage. The scheduler emits sequential single-kernel submissions rather than allowing one launch to overwrite another launch's live state. Whole-graph device compilation still happens before scheduling, so models such as MNIST compile as one batch.

`custom_hcq_runtime_key` is unrelated to lowering: it enables the device-specific runtime cache key, which must include buffer identity and VA for MT's serialized launch lifecycle. Other backends keep the default `None` key behavior.

`MTSubmission` is the named state record for this lifecycle:

- `args` is the runtime-patched argument page, not a captured argument packet.
- `write_map` contains sparse generated pages so the 256 MiB backing view is never cleared or uploaded wholesale.
- `generation` is the MMU generation used by the submission.
- `mmu_pages` snapshots page-table state needed when another launch would otherwise overwrite an earlier shared mapping.
- `grid_deps` is an optional symbolic grid/local-size patch.

Multi-submission/shared-CCB launches retain in-graph completion polling. Only suitable single cached submissions may defer polling to `synchronize()`. Profiled sequential batches are not marked `skip_wait`, because each kernel's hardware timing must be collected.

### Cache identity and lifecycle

The HCQ runtime key includes every input and output `Buffer` object **and its VA**, including outputs. Buffer object identity and VA reuse are not interchangeable. First use always performs the complete launch lifecycle. A cached revisit regenerates state when an actual launch switch requires it; it does not skip first-use activation. Barrier kernels are excluded from the HCQ cache.

## Timing

After completion, profiling reads:

- `launch.v.addr + 0x40a000`: 64-bit hardware timestamp.
- `launch.v.addr + 0x409008`: 32-bit runtime ticks.

End time is `timestamp + runtime`. Values are stored in `signal.index(1)`, matching HCQ2's `2*s+1` profiling convention. `timestamp_divider = 25` corresponds to the 25 MHz timer. Profiled submissions wait for completion before reading these values.

## Validation

- `test/null/test_mt*.py`: mocked ABI, MMU, CCB, PDS, compiler, USB, and launch-layout coverage.
- `test/runtime/test_mt_cache.py`: launch revisits, changing input/output buffers, allocator reuse, symbolic reductions, and profiled revisits.
- `test/runtime/test_ops.py`: complete ops suite, including conv3d spill kernels.
- `examples/beautiful_mnist.py`: whole-graph compilation, sequential launches, training, evaluation, and hardware timing.
