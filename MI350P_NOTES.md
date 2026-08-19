# MI350P (AMD Aqua Vanjaram, PCI 1002:75a8, gfx950) tinygrad/AM bring-up notes

## Status: WORKING
`DEV=PCI+AMD python3 test/test_tiny.py TestTiny.test_plus` → OK (also OK under DEBUG=2).
Full `test/test_tiny.py`: 19/21 pass (2 "errors" are missing `clang` on this host: test_const/test_eye route through a CPU compiler; unrelated to the driver). Repeated runs in the same boot work (partial-boot path functions; boot is ~2.5x faster on re-open).

## Debug method that cracked it
Side-channel watcher (separate root process polling sysfs BAR0 (VRAM) + BAR5 (MMIO) with ms timestamps) showed the window where host access died, first as "PTE already mapped: 0xffffffffffffffff". The ff death came from several distinct bugs layered on top of each other, each indestructible until bisected with env gates (since removed).

## Root causes found (all fixed in-tree, no envs)
1. SDMA init was the fabric killer: tinygrad's `AM_SDMA.init_hw` wrote per-engine `SDMA_CNTL.TRAP_ENABLE` + doorbell route tables. On aqua the SDMA doorbell/context management is firmware/RLC-owned (amdgpu's sdma_v4_4_2 for 4.4.4 never calls `nbio->sdma_doorbell_range`). ~40ms after those writes the chip raised `RAS_ATHUB_ERR_EVENT` (DF ACA), and the host BAR0 view of VRAM died permanently (writes and reads). MMIO survived. Fixed by making `AM_SDMA.init_hw` a no-op on NBIO 7.9.
2. SDMA submission had to be doorbell-free: kernel's own SDMA ring runs with `SDMA_GFX_DOORBELL{,_OFFSET}=0` and submits by writing the `RB_WPTR` register pair. tinygrad now uses an MMIO view of the WPTR register as the queue "doorbell" on NBIO 7.9, and `setup_ring` leaves the doorbell registers alone + rb_priv=0 (kernel values).
3. XCC doorbell fence: kernel writes `regXCC_DOORBELL_FENCE=0xF0` (4 of 8 XCCs harvested); we wrote 0. Wrong fence meant doorbell writes to fenced XCCs produced ATHUB/fabric errors.
4. Harvested XCCs (the dequeue/flush timeouts): discovery lists 8 GC HW instances but the HARV(EST) table marks instances 4-7. Their registers read 0xffffffff; any loop over "xccs=8" hung (`RLC safe-mode timeout`, TLB flush timeout, HQD dequeue timeout) and wrote into phantom space. We now parse the HARVEST table (like `amdgpu_discovery_harvest_ip`) and count only live instances.
5. MEC doorbell layout: aqua uses `DOORBELL_LAYOUT1_MEC_RING_START=8` for the first compute ring doorbell (not NAVI10's 3 = aqua's HIQ slot). Kernel: `aqua_vanjaram_doorbell_index_init`.
6. Boot-ordering/lockouts learned earlier (kept): PSP first, BL-ready exact 0x80000000 with an early ~2s garbage-window tolerance, HDP flush remap (0x1A000) before any flush (silicon default 0x385c is bogus), SPL skipped for MP0 13.0.15, SMU SetDriverDramAddr with any_resp tolerance, EnableAllSmuFeatures skipped (invalid on smu_v13_0_12 family), clock programming skipped (needs full DPM/pptable bring-up), only 2 mmhubs exist host-visible, fb_end computed from vram_size, vmhubs=2, IH rings in sysmem (use_bus_addr semantics) + IH_CHICKEN/+RETRY_INT_CAM, spare TMR AUTOLoad not needed (TMR at boot), spatial-partition cmd skipped on 13.0.15.

## Kernel reference anchors (dkms tree, used for read-only capture)
- /usr/src/amdgpu-6.19.14-2377056.24.04: `aqua_vanjaram.c` (doorbell layout), `nbio_v7_9.c` (HDP remap hole 0x1A000, XCC fence, doorbell-entry programming, ih_doorbell_range), `sdma_v4_4_2.c` (SDMA start/stop, RB_CNTL bit-exact, WPTR submission), `gmc_v9_0.c` (snoop=true for sys PTEs on 9.4.3-9.5.0), `gfx_v9_4_3.c` (RLC safe-mode enter/exit, xcc cp resume), `amdgpu_discovery.c` (harvest table handling).
- Kernel boots the chip from cold state via dkms (`modprobe amdgpu`); after any tinygrad-mode1/timeout the silicon must be power-cycled (BL does not re-POST).

## Known limits / TODO
- RLCS clock programming skipped for MP0 13.0.15 (default clocks; perf tuning would need the full SMU DPM/pptable dance from smu_v13_0_12_ppt.c).
- `IH_CHICKEN`/retry-int-cam writes are aqua-specific (OSSSYS 4.4.2) gated by (9,5) + reg presence.
- AM_RESET/mode1: SMU `GfxDriverReset` leaves the PSP/BL permanently dead on this chip (mailbox registers read 0x0, only power cycle recovers). A full AM re-init therefore distinguishes firmware state: foreign/unknown firmware gets the mode1 attempt (unchanged behavior), while AM-own firmware (SCRATCH_REG7 matches) is re-initialized *without* any reset - the PSP stage is skipped (a live sOS one-shots its ring between sessions - commands written to it are never serviced), and the host-side IP blocks (SOC/GMC/IH/SMU/GFX/SDMA) are fully re-programmed on top of the running firmware. AM_RESET=1 is verified working (also repeatedly).
- `is_hive` tightened to `seg_sz>0 and pf_max_region>0` to avoid misdetecting single-node parts.
- Discovery table is cached per-bus (`~/.cache/tinygrad/downloads/discovery/`) because the top-of-VRAM table becomes firmware-reserved/unreadable after the first boot.
