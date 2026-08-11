# Runbook: Llama 3 8B Training on DigitalOcean MI350X

## Machine Specs
- 8x MI350X GPUs (gfx950, device ID 75b0), 288GB VRAM each
- 2TB RAM, 192 CPUs, 2TB disk
- ROCm 7.14 at `/opt/rocm` (NOT `/opt/rocm-7.1.1` like the submission scripts assume)
- Python 3.12

## Phase 1: System Setup

### 1.1 Install packages
```bash
apt-get update
apt-get install -y python3-pip python3-venv git tmux rclone clang
```

### 1.2 Install Python deps
```bash
python3 -m pip install --break-system-packages numpy tqdm wandb tiktoken sentencepiece
```

### 1.3 Install ROCm dev headers
The base image has ROCm runtime but NOT the HIP dev headers. Need:
```bash
apt-get install -y amdrocm-core-dev
```
This installs `hip/hip_runtime.h` at `/opt/rocm/core-7.14/include/hip/hip_runtime.h`.
The symlink `/opt/rocm/include` → `/opt/rocm/core-7.14/include` makes it available at `/opt/rocm/include/hip/hip_runtime.h`.

### 1.4 Remove Ubuntu comgr, use ROCm comgr
The Ubuntu repo has comgr 6.0 which doesn't know gfx950. Remove it:
```bash
apt-get remove -y libamd-comgr-dev libamd-comgr2
```
ROCm 7.14 ships comgr 3.3 at `/opt/rocm/lib/libamd_comgr.so`. Add ROCm libs to ldconfig:
```bash
cat > /etc/ld.so.conf.d/rocm.conf << 'EOF'
/opt/rocm/lib
/opt/rocm/lib/llvm/lib
/opt/rocm/lib/rocm_sysdeps/lib
EOF
ldconfig
```

### 1.5 Install geohot tmux config
From `geohot/configuration` repo:
```bash
cat > ~/.tmux.conf << 'EOF'
unbind C-b
set -g prefix `
bind-key ` last-window
bind-key e send-prefix

set -g status-position bottom
set -g status-bg colour234
set -g status-fg colour137
set -g status-left ''
set -g status-right '#[fg=colour233,bg=colour241,bold] %d/%m #[fg=colour233,bg=colour245,bold] %H:%M:%S '
set -g status-right-length 50
set -g status-left-length 20
setw -g mode-keys vi

setw -g window-status-current-format ' #I#[fg=colour250]:#[fg=colour255]#W#[fg=colour50]#F '
setw -g window-status-format ' #I#[fg=colour237]:#[fg=colour250]#W#[fg=colour244]#F '

set-option -g history-limit 5000
set -g extended-keys on
set -g extended-keys-format csi-u
EOF
```

### 1.6 Reload amdgpu driver
tinygrad's HCQ backend needs `/dev/kfd` which is created by the amdgpu kernel driver.
If the driver was unloaded, reload it:
```bash
modprobe amdgpu
ls /dev/kfd  # should exist
```

## Phase 2: Clone tinygrad
```bash
cd /root
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install --break-system-packages -e .
```

## Phase 3: Download C4 Dataset

The C4 data is on the MLCommons Cloudflare R2 bucket in Megatron-LM indexed format.

```bash
rclone config create mlc-training s3 provider=Cloudflare \
  access_key_id=76ea42eadb867e854061a1806220ee1e \
  secret_access_key=a53625c4d45e3ca8ac0df8a353ea3a41ffc3292aa25259addd8b7dc5a6ce2936 \
  endpoint=c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com

mkdir -p /root/datasets/c4-8b
rclone copy mlc-training:mlcommons-training-wg-public/llama3_1/datasets/c4/llama3_1_8b/ /root/datasets/c4-8b/ -P
```

Files downloaded (~85GB total, ~6 minutes):
- `c4-train.en_6_text_document.bin` (79 GB)
- `c4-train.en_6_text_document.idx` (870 MB)
- `c4-validation-91205-samples.en_text_document.bin` (159 MB)
- `c4-validation-91205-samples.en_text_document.idx` (1.8 MB)
- `LICENSE.txt`, `NOTICE.txt`

### Symlink for the submission script
The `dev_run.sh` script hardcodes `BASEDIR="/raid/datasets/c4-8b/"`. Symlink:
```bash
mkdir -p /raid/datasets
ln -s /root/datasets/c4-8b /raid/datasets/c4-8b
```

## Phase 4: wandb Login
```bash
wandb login
```
Enter API key from https://wandb.ai/authorize

## Phase 5: Run Training

### 5.1 Smoke test (beam search, 2 layers, fake data)
Always run beam first to validate the pipeline:
```bash
cd /root/tinygrad
COMGR_PATH=/opt/rocm/lib/libamd_comgr.so \
COMGR_3_PATH=/opt/rocm/lib/libamd_comgr.so \
CC=/opt/rocm/core-7.14/lib/llvm/bin/clang \
DEV=AMD:HIP \
ROCM_PATH=/opt/rocm BASEDIR=/root/datasets/c4-8b/ \
  bash examples/mlperf/training_submission_v6.0/tinycorp/benchmarks/llama31_8b/implementations/tinybox_8xMI350X/dev_beam.sh
```

### 5.2 Full training run
```bash
cd /root/tinygrad
COMGR_PATH=/opt/rocm/lib/libamd_comgr.so \
COMGR_3_PATH=/opt/rocm/lib/libamd_comgr.so \
CC=/opt/rocm/core-7.14/lib/llvm/bin/clang \
DEV=AMD:HIP \
ROCM_PATH=/opt/rocm BASEDIR=/root/datasets/c4-8b/ \
WANDB=1 \
  bash examples/mlperf/training_submission_v6.0/tinycorp/benchmarks/llama31_8b/implementations/tinybox_8xMI350X/dev_run.sh
```

## Environment Variable Reference

| Variable | Value | Why |
|---|---|---|
| `COMGR_PATH` | `/opt/rocm/lib/libamd_comgr.so` | tinygrad's DLL loader needs explicit path to find comgr 3.3 |
| `COMGR_3_PATH` | `/opt/rocm/lib/libamd_comgr.so` | comgr 3.x uses a separate `comgr_3` module with its own path var |
| `CC` | `/opt/rocm/core-7.14/lib/llvm/bin/clang` | System clang doesn't know gfx950; must use ROCm's bundled clang |
| `DEV` | `AMD:HIP` | Force HIPRenderer (comgr-based) over HIPCCRenderer (hipcc subprocess) |
| `ROCM_PATH` | `/opt/rocm` | Script defaults to `/opt/rocm-7.1.1` which doesn't exist |
| `BASEDIR` | `/root/datasets/c4-8b/` | Where C4 dataset was downloaded (script hardcodes `/raid/datasets/c4-8b/`) |
| `WANDB` | `1` | Enable wandb logging (off by default) |

## What the submission script sets

The `dev_run.sh` script sets all training-specific env vars:
- `DEV=AMD`, `DP=8`, `MP=1`, `BS=16`, `GRADIENT_ACC_STEPS=2` → GBS=32
- `MXFP4=1`, `ASM_GEMM=1`, `HK_FLASH_ATTENTION=1` — FP4 weights, assembly GEMM, flash attention
- `MASTER_WEIGHTS=1`, `WQKV=1`, `FAST_CE=1` — master weights, fused QKV, fused cross-entropy
- `FUSED_INPUT_QUANTIZE=1`, `FUSED_ADD_NORM_MUL_QUANTIZE=1`, `FUSED_SILU_W13=1` — fused kernels
- `DEFAULT_FLOAT=bfloat16`, `OPTIM_DTYPE=float32`
- `LR=1e-3`, `END_LR=1e-4`, `WARMUP_SAMPLES=4096`, `MAX_STEPS=1200000`
- `SEQLEN=8192`, `LLAMA3_SIZE=8B`, `SMALL=1`
- `JITBEAM=3`, beam search params

## Architecture

| Component | Source file |
|---|---|
| Model | `examples/mlperf/models/flat_llama.py` — FlatTransformer, FP8 MXFP4 weights, fused QKV, flash attention |
| Trainer | `examples/mlperf/model_train.py` → `train_llama3()` |
| Optimizer | `examples/mlperf/optim.py` — GradAccClipAdamW, master weights, FP8 re-quant |
| LR schedule | `examples/mlperf/lr_schedulers.py` — CosineAnnealingLRWithWarmup |
| Dataloader | `examples/mlperf/dataloader.py` — Megatron-LM indexed bin format |
| ASM GEMM | `extra/gemm/cdna_asm_gemm.py` — gfx950 MFMA assembly, MXFP4 |
| Flash attention | `extra/thunder/amd/fa.py` |
| Fused kernels | `extra/llama_kernels/` — rmsnorm, silu, quantize, fused_ce |
| GPU driver | `tinygrad/runtime/ops_amd.py` — HCQ, direct KFD ioctl |
| Renderer | `tinygrad/renderer/cstyle.py` — HIPRenderer for gfx950 |
| comgr compiler | `tinygrad/runtime/support/compiler_amd.py` — HIPCompiler using comgr 3.3 |

## Troubleshooting

### `'hip/hip_runtime.h' file not found`
Install `amdrocm-core-dev`:
```bash
apt-get install -y amdrocm-core-dev
```

### `'gfx950' is not a recognized processor` + LLVM crash
System clang doesn't know gfx950. Set `CC=/opt/rocm/core-7.14/lib/llvm/bin/clang`.

### `comgr not available: try setting COMGR_PATH?`
Remove Ubuntu comgr, add ROCm libs to ldconfig, set `COMGR_PATH` and `COMGR_3_PATH`:
```bash
apt-get remove -y libamd-comgr-dev libamd-comgr2
# add /opt/rocm/lib paths to /etc/ld.so.conf.d/rocm.conf
ldconfig
```

### `comgr not available: try setting COMGR_3_PATH?`
comgr 3.x uses a separate module. Set `COMGR_3_PATH=/opt/rocm/lib/libamd_comgr.so` too.

### `FileNotFoundError: '/raid/datasets/c4-8b/...'`
Script hardcodes `BASEDIR`. Either symlink or edit the script:
```bash
mkdir -p /raid/datasets && ln -s /root/datasets/c4-8b /raid/datasets/c4-8b
```

### `No such file or directory: 'clang'`
Install clang: `apt-get install -y clang` (for CPU compilation).
For gfx950 HIP compilation, comgr (not clang) is used — ensure the ROCm 7.14 comgr 3.3 is properly loaded via `COMGR_PATH` and `COMGR_3_PATH`.

## Appendix: KVM Virtualization Notes

The DigitalOcean MI350X machine is a **KVM virtual machine** with GPU PCI passthrough, not raw metal. This has several implications:

### Virtualization detection
```
$ systemd-detect-virt
kvm
$ lspci -nn | grep AMD
83:00.0 ... Device [1002:75b0]
```
CPU flags include `hypervisor`. `dmesg` shows `Hypervisor detected: KVM`.

### PCI device ID mismatch
The physical GPU has device ID `0x75a0` (visible as the **subsystem** ID), but the KVM host presents it to the guest with device ID `0x75b0`. tinygrad's `PCIIface` in `ops_amd.py` and `hive_reset.py` only listed `0x75a0` (and other IDs), so the GPU was not found.

Fix: add `0x75b0` to the device ID list in:
- `tinygrad/runtime/ops_amd.py` line 845 (`PCIIface.__init__`)
- `extra/amdpci/hive_reset.py` line 14

```python
# ops_amd.py
devices=((0xffff, (0x74a1,0x744c,0x7480,0x7550,0x7551,0x7590,0x75a0,0x75b0)),),
```

### amdgpu driver issues
amdgpu loads and binds to all 8 GPUs, but on this VM it fails to fully initialize:
```
[  799.780369] amdgpu 0000:83:00.0: Failed to alloc msi vectors
[  799.781476] amdgpu 0000:83:00.0: sw_init of IP block <vega20_ih> failed -22
[  799.782724] amdgpu 0000:83:00.0: amdgpu_device_ip_init failed
[  799.793885] amdgpu 0000:83:00.0: Fatal error during GPU init
```
This is because MSI vectors can't be allocated in the VM (no host interrupt remapping exposed to guest). After a failed init, `rmmod amdgpu` can wedge the module (stuck in "Unloading" state), requiring a full VM reboot.

### `/dev/kfd` not usable
Even when amdgpu loads successfully (after a fresh boot), `/dev/kfd` exists but returns `EINVAL` on open. tinygrad's `KFDIface` path cannot be used. The `PCIIface` (direct PCI) path must be used instead.

### VRAM BAR reads all 0xFF
After amdgpu initializes and then is unbound from the GPU, the VRAM BAR reads all `0xFF` — the discovery table at end of VRAM is gone. This means tinygrad's `AMDev._run_discovery()` fails with `discovery signatures mismatch`.

This happens because:
1. amdgpu binds and initializes the GPU (VBIOS populates VRAM)
2. amdgpu is unbound (or fails and finishes)
3. The GPU state is left in a partially cleaned state
4. VRAM contents (including discovery table) are lost

A PCI `reset` (`echo 1 > /sys/bus/pci/devices/0000:83:00.0/reset`) does not repopulate the discovery table — only the VBIOS/firmware boot sequence does.

### VFIO attempt
Tried binding the GPU to `vfio-pci` with `enable_unsafe_noiommu_mode=1`. The GPU binds to vfio-pci successfully, but VRAM BAR still reads all `0xFF`. The GPU needs firmware initialization (by amdgpu) before the discovery table is populated, and after unbinding it's lost.

### No IOMMU in guest
There is no AMD IOMMU visible inside the guest (`dmesg` has no `AMD-Vi` entries). PCI devices have no `iommu_group`. The IOMMU is on the host side, invisible to the guest. This is normal for KVM passthrough.

### No fan control
No `fan*` or `pwm*` hwmon entries are exposed. MI350X fans are managed by the host hardware/BMC, not the guest VM. GPU temps read ~56-63°C, power ~265W each at idle.

### Current status: NOT WORKING
tinygrad's direct PCI path (`PCIIface`) cannot initialize the GPU on this VM because:
1. The VRAM discovery table is not populated (amdgpu must boot the GPU first)
2. After amdgpu unbind, VRAM is cleared
3. tinygrad does not load VBIOS firmware itself — it relies on the discovery table already being in VRAM

On raw metal (e.g. tinybox), the GPU's VBIOS runs at POST time and populates the discovery table before the OS boots. In this KVM VM, the VBIOS does not run, so the discovery table is never populated unless amdgpu initializes it.

Potential paths forward:
- Get amdgpu to fully initialize (fix MSI issue, possibly with `pci=assign-busses` or interrupt remapping on host)
- Use the KFDIface path if `/dev/kfd` can be made to work
- Have tinygrad load VBIOS/firmware itself (like amdgpu does)
