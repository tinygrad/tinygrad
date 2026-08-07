# AM Driver

AM driver is a userspace driver targeting AMD's RDNA3/RDNA4. You only need tinygrad to send compute tasks to your GPU!

## How to run?
Make sure that amdgpu module is unloaded and just run tinygrad with `DEV=AMD`!

Optional requirements:

* vfio-pci module for IRQ handling and IOMMU-protected DMA

When the system IOMMU is enabled (AMD-Vi), the driver must go through vfio so that the GPU's DMA is confined to explicitly
mapped pages: a device fault then hits an IOMMU page fault (and only kills the GPU session) instead of corrupting host memory
and taking the whole system down. This is enabled automatically when the device is behind an IOMMU (set `VFIO=0` to opt out,
e.g. with `iommu=pt`). Note that without an IOMMU (or with `iommu=pt`) DMA is unprotected. P2P between GPUs is only supported
without address translation: boot with `iommu=pt` and set `VFIO=0`.

## Environment Variables

| Variable | Possible Value(s) | Description |
|----------|------------------|-------------|
| AM_RESET | [1] | Performs a full GPU reset (reloading all firmware and IP blocks) |
| AM_DEBUG | [0-4] | Sets the level of additional debugging information |
| VFIO | [0, 1] | Force raw PCI access (0) or vfio (1). By default vfio is used automatically when the device is behind an IOMMU, which requires it |

## AM Driver Details

### Compute & SDMA Queues

AM binds compute queues directly to MEC (bypassing MES). Tinygrad uses only one compute queue, which is bound at `pipe=0 queue=0`. Similarly, the single SDMA queue is bound at `engine=0 queue=0`.

### Boot

The GPU being passed can be in one of several states:
1. Not initialized
2. Initialized by amdgpu
3. Initialized by AM

The first and second states require a full GPU setup since their states are unknown. The second state also requires a mode1 reset to reinitialize all components.

The third state can be set up partially to optimize boot time. In this case, only the GFX and SDMA IPs need to be initialized. To enable this, AM uses a separate boot memory that is guaranteed not to be overwritten. This physical memory is utilized for all blocks that are initialized only during the initial AM boot. To determine if the GPU is in the third state, AM uses `regSCRATCH_REG7` as a flag.

### VM Management

Each AM device sets up only a single `VMID=0` and one page directory. The page directory used is 3-level and thus supports up to 512GB of virtual addresses. All AM devices are located in one virtual address space.