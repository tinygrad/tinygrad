# AMD and NVIDIA GPUs over ASM2464 USB3

This document describes tinygrad's direct USB3 GPU path through custom
ASM2464 firmware. It covers the AMD implementation used as the reference and
the RTX 3090 implementation on the `bounty/nv-usb3-wip` branch. It does not
describe USB4 or Thunderbolt PCIe tunneling: in this path, tinygrad emits PCIe
transactions through a USB vendor protocol without asking the host OS to
enumerate the GPU as a native PCIe device.

The status terms used below are:

- **Qualified**: passed a physical hardware test for the exact configuration
  named here.
- **Implemented**: present in the runtime and covered by source or automated
  tests, but no broader hardware claim is implied.
- **Experimental**: present only in the current uncommitted lifecycle work, or
  observed to work without repeatable qualification.
- **Missing**: no production implementation currently satisfies the required
  behavior.

## The important AMD precedent

AMD does not fully power-cycle the GPU or unload all trusted firmware when a
tinygrad process exits. It defines a reusable warm state instead.

On first boot, tinygrad initializes the security processor (PSP), system
management unit (SMU), trusted memory region (TMR), memory controller, interrupt
handler, graphics block, and SDMA. Boot-only allocations come from a reserved
32 MiB VRAM region. On clean close, tinygrad dequeues the GFX queues, disables
the SDMA rings, soft-resets SDMA where supported, lowers clocks, drains
interrupts, and records a clean marker. PSP/sOS, SMU, TMR, and the boot
allocations remain resident.

The next process recognizes tinygrad's version marker and clean-close marker,
reuses the retained base state, and only reinitializes GFX and SDMA. If the
previous process did not close cleanly, or a GC fault remains, tinygrad rejects
that partial path and performs full initialization, using an SMU mode-1 reset
first when PSP and SMU are still alive.

This is the relevant design lesson for NVIDIA: reliable reuse requires a
documented state at the process boundary. It does not inherently require the
GPU to look electrically cold.

## Shared ASM2464 transport

Both backends use the same userspace PCIe bridge support in
[`tinygrad/runtime/support/system.py`](../../tinygrad/runtime/support/system.py)
and [`tinygrad/runtime/support/usb.py`](../../tinygrad/runtime/support/usb.py):

1. Enumerate custom firmware at USB IDs `add1:0001` or `3801:0001`.
2. Power the PCIe side of the ASM2464 when needed and wait for link state L0.
3. Walk the bridge hierarchy with PCIe configuration TLPs until the endpoint is
   found, checking for the expected AMD or NVIDIA vendor ID.
4. Configure bridge windows, resize and assign endpoint BARs, and enable PCIe
   bus mastering.
5. Represent config space and BAR mappings with `USBMMIOInterface`, which turns
   reads and writes into custom firmware requests.

The custom firmware exposes three important mechanisms:

- `F0` emits PCIe config and memory TLPs. Its bulk modes provide aligned PCIe
  memory reads and writes.
- `F2` moves bulk data between the host and the ASM2464's 512 KiB SRAM. The GPU
  sees that SRAM beginning at system address `0x200000`.
- `F3` controls PCIe endpoint power.

The firmware source is maintained separately in
[`tinygrad/asm2464pd-firmware`](https://github.com/tinygrad/asm2464pd-firmware).
AMD currently relies on the commands operationally but does not request the
`F4` protocol-version record. NVIDIA requires protocol 1.x, firmware revision 4
or newer, and the capabilities used by its boot path.

## AMD implementation

### Device discovery and boot

`DEV=USB+AMD` selects `USBIface` in
[`tinygrad/runtime/ops_amd.py`](../../tinygrad/runtime/ops_amd.py). After the
shared bridge setup, `AMDev` maps the VRAM, doorbell, and register BARs and
reads AMD's discovery table to select the correct register and firmware layout
for the installed ASIC.

The driver supports its AM architecture set over this interface: gfx9.4.2,
gfx9.5.0, and gfx major versions 11 and 12. That is a source-level support
statement, not a claim that every such board has been physically tested over
every ASM2464 enclosure.

The boot decision in
[`tinygrad/runtime/support/am/amdev.py`](../../tinygrad/runtime/support/am/amdev.py)
has two paths:

- **Full boot** is used for an unknown GPU state, `AM_RESET=1`, a previous
  unclean exit, or a pending GC fault. If PSP and SMU are already alive,
  tinygrad first performs an SMU mode-1 reset. It then initializes SOC, GMC,
  IH, PSP, SMU, GFX, and SDMA.
- **Partial boot** is used when `SCRATCH_REG7` contains the current AM driver
  version and `SCRATCH_REG6` says the prior process finalized cleanly. The
  retained management firmware and boot memory are reused. Tinygrad resets and
  configures MEC, then initializes fresh GFX and SDMA queues.

`SCRATCH_REG6` is set to 1 while a process owns the device. Clean finalization
writes 0; an error writes 1. A crash normally leaves 1 in place, which causes
the next process to take the full recovery path rather than trust stale queue
state.

### AMD firmware loading

`AMFirmware` fetches hash-pinned PSP/sOS, SMU, SDMA, MEC/PFP/ME, IMU, and RLC
images. On USB, PSP's message-1 staging area is the controller's 512 KiB SRAM,
mapped into the GPU's system address space. Tinygrad puts one component in that
area, tells the PSP to authenticate and consume it, waits for completion, and
then reuses the same area for the next component. It is ordinary sequential
staging; AMD does not need the timing-dependent cyclic page mapping used for
the much larger NVIDIA GSP image.

Once sOS is running, PSP creates its command ring, establishes or loads TMR,
and loads the remaining IP firmware. A clean partial boot skips this firmware
load because those components and their reserved boot allocations remain in
the reusable state.

### Memory and transfers

AMD creates one three-level GPU page-directory hierarchy at `VMID=0`, with up
to 512 GiB of virtual address space. Normal allocations and page tables live in
VRAM. The USB interface reserves three controller-visible regions:

- `0x200000`, 512 KiB SRAM, used as the host/GPU copy staging buffer and PSP
  message-1 buffer.
- `0x820000`, 4 KiB, used for small host-visible control allocations.
- `0x822000`, 4 KiB, used for copy completion and controller handshakes.

Host-to-GPU copies are split into staging-buffer-sized pieces. The host writes
each piece to SRAM over USB, and the AMD SDMA queue copies it into its final
VRAM allocation. GPU-to-host copies reverse that operation: tinygrad arms an
`F2` bulk read, SDMA copies the next piece from VRAM into SRAM, and a GPU write
to the completion region tells the controller that the data is ready for the
host.

Direct peer-device transfer is disabled for USB devices. USB also uses smaller
fixed resources than native PCI: an 8 KiB compute ring, a 512-byte SDMA ring,
8 KiB of kernel arguments, and 256 bytes of signals. USB submission does not
use the native PCI busy-wait for SDMA ring space, and its write logic explicitly
handles commands that would wrap the small ring.

### Compute and synchronization

Tinygrad binds a compute queue directly to MEC, normally pipe 0 queue 0, and an
SDMA queue to engine 0 queue 0. PM4 or AQL packets are written to the ring and
submitted through MMIO doorbells. The same HIP, LLVM, and HIPCC renderer
choices used by the AM backend are available; the USB transport changes device
access, not tinygrad's kernel model.

The USB path has no native interrupt file descriptor, so its interface sleep
hook is a no-op and normal completion is observed through timeline values.
MEC reset/recovery code exists in the shared AM driver, but USB does not perform
the native PCI interrupt-drain path during each synchronization. Consequently,
compute and timeout recovery are implemented, while native-equivalent USB fault
reporting should not be inferred.

### Close, reopen, and power

On close, `AMDev.fini` disables SDMA rings, dequeues GFX hardware queues,
soft-resets SDMA where supported, selects the SMU's lowest clock level, drains
pending IH events, and writes the finalized-state marker. It does not turn off
PCIe power, destroy TMR, or unload PSP/sOS.

This produces three intentional next-open cases:

| Observed state | AMD action |
|---|---|
| No tinygrad version marker | Full initialization |
| Version marker plus clean finalization and no GC fault | Reuse firmware and boot memory; reset/reinitialize GFX and SDMA |
| Version marker plus unclean finalization or GC fault | Reject partial boot; use mode-1 reset when PSP/SMU are alive, then fully initialize |

The low-clock close state is an idle state, not D3cold and not zero power.

### AMD verification

The repository exercises the implementation at two levels:

- `GMMU=0 DEV=MOCKUSB+AMD python test/test_tiny.py` runs in regular CI against
  the modeled controller, SRAM windows, PCI topology, BARs, queues, and copies.
- The tracked USB GPU benchmark job runs `TestTiny.test_plus`, the full
  `test/test_tiny.py`, and
  [`test/external/external_test_usb_asm24.py`](../../test/external/external_test_usb_asm24.py)
  on physical AMD USB3 hardware. The external test checks exact random-data
  round trips in addition to reporting copy speed.

## NVIDIA RTX 3090 implementation

### Qualified cold-to-compute path

`DEV=USB+NV:NAK` selects the NVIDIA USB interface in
[`tinygrad/runtime/ops_nv.py`](../../tinygrad/runtime/ops_nv.py). The committed
branch has physically reached tinygrad compute on an RTX 3090 (`10de:2204`)
through an ADT-Link UT3G V1.6 at 5 Gbit/s SuperSpeed USB. The qualification gate
checked:

- USB and PCI identities, SuperSpeed link, and BAR setup.
- The exact result `[4, 7, 10, 13]` from a GPU tensor expression.
- An exact 4 MiB host-to-GPU-to-host round trip.
- No configured compute or DMA channel fault bits and zero visible PCIe AER
  status.

Earlier boundary tests also passed exact 64, 128, and 256 MiB round trips. A
historical two-fresh-process run passed, but later lifecycle probing found that
fresh-process restart is not repeatable. It therefore qualifies cold boot,
compute, and transfer on the named hardware, not reliable teardown/reopen.

Observed initialization was about 17 seconds. The 37 to 39 second qualification
worker additionally performed tensor setup, compute, the exact transfer, and
channel/AER health checks; that full worker time is not the boot time.

### Controller firmware

The NVIDIA USB path assumes the matching current ASM2464 firmware source. The
host does not negotiate provisional firmware revisions or preserve compatibility
with earlier experimental images.

### GSP boot without host RAM

The difficult NVIDIA-specific problem is the roughly 63 MiB signed GSP image.
The normal GSP boot ABI expects host-addressable pages, but this direct USB3
path has no general GPU DMA mapping of host RAM. Only the controller's 512 KiB
SRAM is GPU-visible as system memory.

The implementation in
[`tinygrad/runtime/support/nv/ip.py`](../../tinygrad/runtime/support/nv/ip.py)
builds the required WPR metadata, signature, bootloader, radix page tables, and
an 84-page image window in that SRAM. The page tables describe the complete GSP
image but cycle its logical pages over those 84 physical SRAM pages. While SEC2
verifies and consumes the image, the host refills the window with successive
28-page batches. The PCIe link is temporarily forced to Gen1, ASPM is disabled,
and read-request size is reduced so the measured refill schedule remains ahead
of SEC2. This is PCIe Gen1 behind a SuperSpeed USB connection; it is not USB 1.

After verification, the GSP image resides in the GPU's protected VRAM region
and normal runtime traffic no longer uses that timed image stream. The GSP RPC
queues still need their expected system-memory layout, so tinygrad maps their
logical pages across SRAM and the controller's small XDATA windows. Other boot
structures are allocated in the 256 MiB CPU-visible BAR1 region of VRAM.

### NVIDIA memory, compute, and transfers

Unlike AMD's `F2`/SDMA data path, normal NVIDIA tensor copies use BAR1-visible
VRAM staging buffers. The host bulk-writes or bulk-reads those buffers with the
firmware's PCIe memory transport, and NVIDIA's copy GPFIFO moves data between
staging and final VRAM allocations. Triple buffering is used because BAR1 is a
fixed 256 MiB aperture shared with boot and kernel-argument allocations.

GSP/RM creates the virtual address space, channel group, compute GPFIFO, DMA
GPFIFO, and graphics context. Ampere and Ada submission uses the engine's
internal runlist doorbell and the channel ID portion of RM's submit token. The
NAK renderer was used for qualification, so no CUDA toolkit was required.

### Current lifecycle status

The committed NVIDIA path sends `UNLOADING_GUEST_DRIVER` on close. It does not
complete the full pre-Hopper sequence that shuts down FWSEC, executes Booter
Unload, and proves that WPR2 is gone. `NVDev` also has no AMD-like retained boot
state: it creates a new GSP client on every process open.

The current uncommitted worktree contains an experimental implementation of the
official full-unload shape:

1. Ask GSP/RM to unload and wait for processor-suspended state.
2. Run FWSEC shutdown.
3. Run the signed Booter Unload image.
4. Require WPR2 to read as cleared.
5. Hot-reset the endpoint and rebuild PCIe/BAR configuration.

This sequence produced two successful independent compute lifecycles during
probing, then a later Booter Load failed with mailbox `0x0b`. That code is not
qualified. In particular, its broad "catch any boot exception, tear down, and
retry once" behavior does not identify a valid state transition and is not a
production recovery policy.

The worktree also clears RM's performance boost after synchronization and
restores it on the next submission. That is an in-process low-power experiment,
not GSP shutdown, PCIe power-down, or a solution for fresh-process reuse.

## Capability comparison

| Capability | AMD over ASM2464 USB3 | NVIDIA RTX 3090 over ASM2464 USB3 | Parity status |
|---|---|---|---|
| USB enumeration | Custom IDs `add1:0001` and `3801:0001` | Same IDs | Implemented |
| Direct userspace PCIe | `F0` config/memory TLPs through `USBMMIOInterface` | Same shared transport | Implemented |
| PCI topology and BAR setup | Bridge discovery, windows, BAR assignment, bus mastering | Same, with fixed 256 MiB BAR1 constraints | Implemented |
| Firmware compatibility check | Assumes matching current firmware | Assumes matching current firmware | Same policy |
| Reproducible controller firmware | Public firmware source provides the AMD path | Current firmware source is physically qualified | Implemented |
| GPU identification | AMD discovery table selects IP versions | PCI ID plus GA/AD/GB register selection; RTX 3090 physically tested | Implemented; NVIDIA qualification is model-specific |
| Trusted firmware source | Hash-pinned AMD PSP, SMU, SDMA, GFX firmware | Hash-pinned NVIDIA 570.144 GSP, bootloader, and security firmware | Implemented |
| Firmware staging | Reuses 512 KiB SRAM for one PSP-consumed component at a time | Timed cyclic page mapping streams the roughly 63 MiB GSP through SRAM | Implemented; NVIDIA is timing-sensitive |
| Protected firmware region | PSP establishes/loads TMR | FWSEC establishes WPR2; Booter Load authenticates and starts GSP | Implemented for cold boot |
| Management firmware after close | Intentionally retains PSP/sOS, SMU, and TMR | Current architecture starts a new GSP client each process | Different lifecycle models |
| Boot allocations | Reserved 32 MiB VRAM region survives partial boot | Boot and teardown data use ordinary CPU-visible VRAM allocations | Implemented |
| Virtual memory | Driver-owned VMID 0 and three-level page directory | GSP/RM-assisted address space and tinygrad page tables | Implemented |
| Compute queue | Direct MEC queue, PM4 or AQL | RM-created compute GPFIFO | Implemented and physically exercised |
| Copy queue | Direct SDMA ring | RM-created DMA GPFIFO | Implemented and physically exercised |
| Host-to-device copy | USB to SRAM, then SDMA to VRAM | USB PCIe write to BAR1 staging, then DMA to VRAM | Implemented |
| Device-to-host copy | SDMA to SRAM, completion trigger, USB bulk read | DMA to BAR1 staging, then USB PCIe read | Implemented |
| Exact transfer testing | Random exact round trip; tracked physical job and mock path | Exact 4 MiB qualification and historical 64/128/256 MiB gates | Qualified on tested hardware |
| Kernel execution | HIP, LLVM, or HIPCC renderer through HCQ | NAK qualification; CUDA/PTX/NVCC renderers exist in backend | Qualified for tested paths |
| Clean process close | Remove GFX/SDMA queues, lower clocks, write clean marker | Free RM client, suspend GSP, run FWSEC shutdown and Booter Unload, verify WPR2 clear | Implemented and physically qualified |
| Clean fresh-process reopen | Reuse retained firmware and boot memory; reinitialize GFX/SDMA | Start from GFW-complete, WPR2-clear state and perform a normal fresh GSP boot | Implemented and physically qualified |
| Unclean-exit detection | Ownership marker and GC fault state force full recovery | Active WPR2 is detectable | Detection implemented |
| Unclean-exit recovery | Reject partial boot; use mode-1 reset when PSP/SMU are alive, then fully initialize | Require advertised PCIe FLR, issue one FLR, verify GFW and WPR2, then fully initialize | Implemented and physically qualified on RTX 3090 |
| Full secure teardown | Not required by AMD's retained-state contract | FWSEC shutdown plus Booter Unload clears WPR2 before close | Implemented and physically qualified |
| Idle power after work | Clean close selects the lowest SMU clock while firmware stays resident | Worktree clears RM perf boost in process; no qualified close-state policy | NVIDIA experimental |
| Fault reporting | Timeline completion; shared MEC recovery, but reduced USB interrupt handling | GSP/RPC, channel-state, timeout, and AER checks | Basic checks exist; neither is native-PCI equivalent over USB |
| Direct peer transfer | Disabled for USB | Not qualified/supported for this path | No parity requirement for single-GPU goal |
| Automated regression | Full `MOCKUSB+AMD` tiny test plus physical benchmark workflow | Focused USB/GSP/unit tests plus clean-reopen and SIGKILL-to-FLR physical gates | Implemented for the tested RTX 3090 path |

## NVIDIA lifecycle contract

The normal process boundary uses full secure teardown: free the RM client,
suspend GSP, run FWSEC shutdown and Booter Unload, verify WPR2 is clear, and
then close. The next process performs an ordinary fresh boot.

If a process dies while WPR2 is active, the next process does not reconnect to
the abandoned GSP queues or consume persisted process state. It instead:

1. Requires the endpoint to advertise PCIe Function Level Reset (FLR).
2. Disables bus mastering and waits for pending PCIe transactions to drain.
3. Issues FLR once, waits for the same PCI identity to return, and rebuilds the
   USB bridge BAR configuration.
4. Remaps BAR0, waits for GA102 GFW completion, and requires WPR2 to be clear.
5. Continues with a normal fresh boot.

There is no catch-all boot retry and no Secondary Bus Reset fallback in this
path. A failed FLR or failed postcondition stops initialization in the unclean
recovery phase without running teardown against partially reset state.

This exact SIGKILL-to-FLR sequence is physically qualified on the RTX 3090 with
trivial compute, an exact 4 MiB transfer, clean teardown, and final WPR2,
and PCIe AER checks. Remaining lifecycle work is broader GPU model qualification
and publication of the current controller firmware source used by the
physical gate.

## Source map

- Shared USB and PCIe transport:
  [`tinygrad/runtime/support/usb.py`](../../tinygrad/runtime/support/usb.py),
  [`tinygrad/runtime/support/system.py`](../../tinygrad/runtime/support/system.py)
- AMD runtime and lifecycle:
  [`tinygrad/runtime/ops_amd.py`](../../tinygrad/runtime/ops_amd.py),
  [`tinygrad/runtime/support/am/amdev.py`](../../tinygrad/runtime/support/am/amdev.py),
  [`tinygrad/runtime/support/am/ip.py`](../../tinygrad/runtime/support/am/ip.py)
- NVIDIA runtime and GSP boot:
  [`tinygrad/runtime/ops_nv.py`](../../tinygrad/runtime/ops_nv.py),
  [`tinygrad/runtime/support/nv/nvdev.py`](../../tinygrad/runtime/support/nv/nvdev.py),
  [`tinygrad/runtime/support/nv/ip.py`](../../tinygrad/runtime/support/nv/ip.py)
- Physical tests:
  [`test/external/external_test_usb_asm24.py`](../../test/external/external_test_usb_asm24.py),
  [`test/external/external_test_nv_usb3.py`](../../test/external/external_test_nv_usb3.py),
  [`test/external/external_test_nv_usb_lifecycle.py`](../../test/external/external_test_nv_usb_lifecycle.py)
- NVIDIA reference teardown sequence:
  [`kernel_gsp.c`](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/610.43.03/src/nvidia/src/kernel/gpu/gsp/kernel_gsp.c),
  [`kernel_gsp_tu102.c`](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/610.43.03/src/nvidia/src/kernel/gpu/gsp/arch/turing/kernel_gsp_tu102.c),
  [`kernel_gsp_booter_tu102.c`](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/610.43.03/src/nvidia/src/kernel/gpu/gsp/arch/turing/kernel_gsp_booter_tu102.c)
