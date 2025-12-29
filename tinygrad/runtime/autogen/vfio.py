# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import Array, DLL, Pointer, Struct, Union, field, CEnum, _IO, _IOW, _IOR, _IOWR
class struct_vfio_info_cap_header(Struct): pass
__u16 = ctypes.c_uint16
__u32 = ctypes.c_uint32
struct_vfio_info_cap_header.SIZE = 8
struct_vfio_info_cap_header._fields_ = ['id', 'version', 'next']
setattr(struct_vfio_info_cap_header, 'id', field(0, ctypes.c_uint16))
setattr(struct_vfio_info_cap_header, 'version', field(2, ctypes.c_uint16))
setattr(struct_vfio_info_cap_header, 'next', field(4, ctypes.c_uint32))
class struct_vfio_group_status(Struct): pass
struct_vfio_group_status.SIZE = 8
struct_vfio_group_status._fields_ = ['argsz', 'flags']
setattr(struct_vfio_group_status, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_group_status, 'flags', field(4, ctypes.c_uint32))
class struct_vfio_device_info(Struct): pass
struct_vfio_device_info.SIZE = 24
struct_vfio_device_info._fields_ = ['argsz', 'flags', 'num_regions', 'num_irqs', 'cap_offset', 'pad']
setattr(struct_vfio_device_info, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_device_info, 'flags', field(4, ctypes.c_uint32))
setattr(struct_vfio_device_info, 'num_regions', field(8, ctypes.c_uint32))
setattr(struct_vfio_device_info, 'num_irqs', field(12, ctypes.c_uint32))
setattr(struct_vfio_device_info, 'cap_offset', field(16, ctypes.c_uint32))
setattr(struct_vfio_device_info, 'pad', field(20, ctypes.c_uint32))
class struct_vfio_device_info_cap_pci_atomic_comp(Struct): pass
struct_vfio_device_info_cap_pci_atomic_comp.SIZE = 16
struct_vfio_device_info_cap_pci_atomic_comp._fields_ = ['header', 'flags', 'reserved']
setattr(struct_vfio_device_info_cap_pci_atomic_comp, 'header', field(0, struct_vfio_info_cap_header))
setattr(struct_vfio_device_info_cap_pci_atomic_comp, 'flags', field(8, ctypes.c_uint32))
setattr(struct_vfio_device_info_cap_pci_atomic_comp, 'reserved', field(12, ctypes.c_uint32))
class struct_vfio_region_info(Struct): pass
__u64 = ctypes.c_uint64
struct_vfio_region_info.SIZE = 32
struct_vfio_region_info._fields_ = ['argsz', 'flags', 'index', 'cap_offset', 'size', 'offset']
setattr(struct_vfio_region_info, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_region_info, 'flags', field(4, ctypes.c_uint32))
setattr(struct_vfio_region_info, 'index', field(8, ctypes.c_uint32))
setattr(struct_vfio_region_info, 'cap_offset', field(12, ctypes.c_uint32))
setattr(struct_vfio_region_info, 'size', field(16, ctypes.c_uint64))
setattr(struct_vfio_region_info, 'offset', field(24, ctypes.c_uint64))
class struct_vfio_region_sparse_mmap_area(Struct): pass
struct_vfio_region_sparse_mmap_area.SIZE = 16
struct_vfio_region_sparse_mmap_area._fields_ = ['offset', 'size']
setattr(struct_vfio_region_sparse_mmap_area, 'offset', field(0, ctypes.c_uint64))
setattr(struct_vfio_region_sparse_mmap_area, 'size', field(8, ctypes.c_uint64))
class struct_vfio_region_info_cap_sparse_mmap(Struct): pass
struct_vfio_region_info_cap_sparse_mmap.SIZE = 16
struct_vfio_region_info_cap_sparse_mmap._fields_ = ['header', 'nr_areas', 'reserved', 'areas']
setattr(struct_vfio_region_info_cap_sparse_mmap, 'header', field(0, struct_vfio_info_cap_header))
setattr(struct_vfio_region_info_cap_sparse_mmap, 'nr_areas', field(8, ctypes.c_uint32))
setattr(struct_vfio_region_info_cap_sparse_mmap, 'reserved', field(12, ctypes.c_uint32))
setattr(struct_vfio_region_info_cap_sparse_mmap, 'areas', field(16, Array(struct_vfio_region_sparse_mmap_area, 0)))
class struct_vfio_region_info_cap_type(Struct): pass
struct_vfio_region_info_cap_type.SIZE = 16
struct_vfio_region_info_cap_type._fields_ = ['header', 'type', 'subtype']
setattr(struct_vfio_region_info_cap_type, 'header', field(0, struct_vfio_info_cap_header))
setattr(struct_vfio_region_info_cap_type, 'type', field(8, ctypes.c_uint32))
setattr(struct_vfio_region_info_cap_type, 'subtype', field(12, ctypes.c_uint32))
class struct_vfio_region_gfx_edid(Struct): pass
struct_vfio_region_gfx_edid.SIZE = 24
struct_vfio_region_gfx_edid._fields_ = ['edid_offset', 'edid_max_size', 'edid_size', 'max_xres', 'max_yres', 'link_state']
setattr(struct_vfio_region_gfx_edid, 'edid_offset', field(0, ctypes.c_uint32))
setattr(struct_vfio_region_gfx_edid, 'edid_max_size', field(4, ctypes.c_uint32))
setattr(struct_vfio_region_gfx_edid, 'edid_size', field(8, ctypes.c_uint32))
setattr(struct_vfio_region_gfx_edid, 'max_xres', field(12, ctypes.c_uint32))
setattr(struct_vfio_region_gfx_edid, 'max_yres', field(16, ctypes.c_uint32))
setattr(struct_vfio_region_gfx_edid, 'link_state', field(20, ctypes.c_uint32))
class struct_vfio_device_migration_info(Struct): pass
struct_vfio_device_migration_info.SIZE = 32
struct_vfio_device_migration_info._fields_ = ['device_state', 'reserved', 'pending_bytes', 'data_offset', 'data_size']
setattr(struct_vfio_device_migration_info, 'device_state', field(0, ctypes.c_uint32))
setattr(struct_vfio_device_migration_info, 'reserved', field(4, ctypes.c_uint32))
setattr(struct_vfio_device_migration_info, 'pending_bytes', field(8, ctypes.c_uint64))
setattr(struct_vfio_device_migration_info, 'data_offset', field(16, ctypes.c_uint64))
setattr(struct_vfio_device_migration_info, 'data_size', field(24, ctypes.c_uint64))
class struct_vfio_region_info_cap_nvlink2_ssatgt(Struct): pass
struct_vfio_region_info_cap_nvlink2_ssatgt.SIZE = 16
struct_vfio_region_info_cap_nvlink2_ssatgt._fields_ = ['header', 'tgt']
setattr(struct_vfio_region_info_cap_nvlink2_ssatgt, 'header', field(0, struct_vfio_info_cap_header))
setattr(struct_vfio_region_info_cap_nvlink2_ssatgt, 'tgt', field(8, ctypes.c_uint64))
class struct_vfio_region_info_cap_nvlink2_lnkspd(Struct): pass
struct_vfio_region_info_cap_nvlink2_lnkspd.SIZE = 16
struct_vfio_region_info_cap_nvlink2_lnkspd._fields_ = ['header', 'link_speed', '__pad']
setattr(struct_vfio_region_info_cap_nvlink2_lnkspd, 'header', field(0, struct_vfio_info_cap_header))
setattr(struct_vfio_region_info_cap_nvlink2_lnkspd, 'link_speed', field(8, ctypes.c_uint32))
setattr(struct_vfio_region_info_cap_nvlink2_lnkspd, '__pad', field(12, ctypes.c_uint32))
class struct_vfio_irq_info(Struct): pass
struct_vfio_irq_info.SIZE = 16
struct_vfio_irq_info._fields_ = ['argsz', 'flags', 'index', 'count']
setattr(struct_vfio_irq_info, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_irq_info, 'flags', field(4, ctypes.c_uint32))
setattr(struct_vfio_irq_info, 'index', field(8, ctypes.c_uint32))
setattr(struct_vfio_irq_info, 'count', field(12, ctypes.c_uint32))
class struct_vfio_irq_set(Struct): pass
__u8 = ctypes.c_ubyte
struct_vfio_irq_set.SIZE = 20
struct_vfio_irq_set._fields_ = ['argsz', 'flags', 'index', 'start', 'count', 'data']
setattr(struct_vfio_irq_set, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_irq_set, 'flags', field(4, ctypes.c_uint32))
setattr(struct_vfio_irq_set, 'index', field(8, ctypes.c_uint32))
setattr(struct_vfio_irq_set, 'start', field(12, ctypes.c_uint32))
setattr(struct_vfio_irq_set, 'count', field(16, ctypes.c_uint32))
setattr(struct_vfio_irq_set, 'data', field(20, Array(ctypes.c_ubyte, 0)))
_anonenum0 = CEnum(ctypes.c_uint32)
VFIO_PCI_BAR0_REGION_INDEX = _anonenum0.define('VFIO_PCI_BAR0_REGION_INDEX', 0)
VFIO_PCI_BAR1_REGION_INDEX = _anonenum0.define('VFIO_PCI_BAR1_REGION_INDEX', 1)
VFIO_PCI_BAR2_REGION_INDEX = _anonenum0.define('VFIO_PCI_BAR2_REGION_INDEX', 2)
VFIO_PCI_BAR3_REGION_INDEX = _anonenum0.define('VFIO_PCI_BAR3_REGION_INDEX', 3)
VFIO_PCI_BAR4_REGION_INDEX = _anonenum0.define('VFIO_PCI_BAR4_REGION_INDEX', 4)
VFIO_PCI_BAR5_REGION_INDEX = _anonenum0.define('VFIO_PCI_BAR5_REGION_INDEX', 5)
VFIO_PCI_ROM_REGION_INDEX = _anonenum0.define('VFIO_PCI_ROM_REGION_INDEX', 6)
VFIO_PCI_CONFIG_REGION_INDEX = _anonenum0.define('VFIO_PCI_CONFIG_REGION_INDEX', 7)
VFIO_PCI_VGA_REGION_INDEX = _anonenum0.define('VFIO_PCI_VGA_REGION_INDEX', 8)
VFIO_PCI_NUM_REGIONS = _anonenum0.define('VFIO_PCI_NUM_REGIONS', 9)

_anonenum1 = CEnum(ctypes.c_uint32)
VFIO_PCI_INTX_IRQ_INDEX = _anonenum1.define('VFIO_PCI_INTX_IRQ_INDEX', 0)
VFIO_PCI_MSI_IRQ_INDEX = _anonenum1.define('VFIO_PCI_MSI_IRQ_INDEX', 1)
VFIO_PCI_MSIX_IRQ_INDEX = _anonenum1.define('VFIO_PCI_MSIX_IRQ_INDEX', 2)
VFIO_PCI_ERR_IRQ_INDEX = _anonenum1.define('VFIO_PCI_ERR_IRQ_INDEX', 3)
VFIO_PCI_REQ_IRQ_INDEX = _anonenum1.define('VFIO_PCI_REQ_IRQ_INDEX', 4)
VFIO_PCI_NUM_IRQS = _anonenum1.define('VFIO_PCI_NUM_IRQS', 5)

_anonenum2 = CEnum(ctypes.c_uint32)
VFIO_CCW_CONFIG_REGION_INDEX = _anonenum2.define('VFIO_CCW_CONFIG_REGION_INDEX', 0)
VFIO_CCW_NUM_REGIONS = _anonenum2.define('VFIO_CCW_NUM_REGIONS', 1)

_anonenum3 = CEnum(ctypes.c_uint32)
VFIO_CCW_IO_IRQ_INDEX = _anonenum3.define('VFIO_CCW_IO_IRQ_INDEX', 0)
VFIO_CCW_CRW_IRQ_INDEX = _anonenum3.define('VFIO_CCW_CRW_IRQ_INDEX', 1)
VFIO_CCW_REQ_IRQ_INDEX = _anonenum3.define('VFIO_CCW_REQ_IRQ_INDEX', 2)
VFIO_CCW_NUM_IRQS = _anonenum3.define('VFIO_CCW_NUM_IRQS', 3)

_anonenum4 = CEnum(ctypes.c_uint32)
VFIO_AP_REQ_IRQ_INDEX = _anonenum4.define('VFIO_AP_REQ_IRQ_INDEX', 0)
VFIO_AP_NUM_IRQS = _anonenum4.define('VFIO_AP_NUM_IRQS', 1)

class struct_vfio_pci_dependent_device(Struct): pass
struct_vfio_pci_dependent_device.SIZE = 8
struct_vfio_pci_dependent_device._fields_ = ['group_id', 'devid', 'segment', 'bus', 'devfn']
setattr(struct_vfio_pci_dependent_device, 'group_id', field(0, ctypes.c_uint32))
setattr(struct_vfio_pci_dependent_device, 'devid', field(0, ctypes.c_uint32))
setattr(struct_vfio_pci_dependent_device, 'segment', field(4, ctypes.c_uint16))
setattr(struct_vfio_pci_dependent_device, 'bus', field(6, ctypes.c_ubyte))
setattr(struct_vfio_pci_dependent_device, 'devfn', field(7, ctypes.c_ubyte))
class struct_vfio_pci_hot_reset_info(Struct): pass
struct_vfio_pci_hot_reset_info.SIZE = 12
struct_vfio_pci_hot_reset_info._fields_ = ['argsz', 'flags', 'count', 'devices']
setattr(struct_vfio_pci_hot_reset_info, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_pci_hot_reset_info, 'flags', field(4, ctypes.c_uint32))
setattr(struct_vfio_pci_hot_reset_info, 'count', field(8, ctypes.c_uint32))
setattr(struct_vfio_pci_hot_reset_info, 'devices', field(12, Array(struct_vfio_pci_dependent_device, 0)))
class struct_vfio_pci_hot_reset(Struct): pass
__s32 = ctypes.c_int32
struct_vfio_pci_hot_reset.SIZE = 12
struct_vfio_pci_hot_reset._fields_ = ['argsz', 'flags', 'count', 'group_fds']
setattr(struct_vfio_pci_hot_reset, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_pci_hot_reset, 'flags', field(4, ctypes.c_uint32))
setattr(struct_vfio_pci_hot_reset, 'count', field(8, ctypes.c_uint32))
setattr(struct_vfio_pci_hot_reset, 'group_fds', field(12, Array(ctypes.c_int32, 0)))
class struct_vfio_device_gfx_plane_info(Struct): pass
struct_vfio_device_gfx_plane_info.SIZE = 64
struct_vfio_device_gfx_plane_info._fields_ = ['argsz', 'flags', 'drm_plane_type', 'drm_format', 'drm_format_mod', 'width', 'height', 'stride', 'size', 'x_pos', 'y_pos', 'x_hot', 'y_hot', 'region_index', 'dmabuf_id', 'reserved']
setattr(struct_vfio_device_gfx_plane_info, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_device_gfx_plane_info, 'flags', field(4, ctypes.c_uint32))
setattr(struct_vfio_device_gfx_plane_info, 'drm_plane_type', field(8, ctypes.c_uint32))
setattr(struct_vfio_device_gfx_plane_info, 'drm_format', field(12, ctypes.c_uint32))
setattr(struct_vfio_device_gfx_plane_info, 'drm_format_mod', field(16, ctypes.c_uint64))
setattr(struct_vfio_device_gfx_plane_info, 'width', field(24, ctypes.c_uint32))
setattr(struct_vfio_device_gfx_plane_info, 'height', field(28, ctypes.c_uint32))
setattr(struct_vfio_device_gfx_plane_info, 'stride', field(32, ctypes.c_uint32))
setattr(struct_vfio_device_gfx_plane_info, 'size', field(36, ctypes.c_uint32))
setattr(struct_vfio_device_gfx_plane_info, 'x_pos', field(40, ctypes.c_uint32))
setattr(struct_vfio_device_gfx_plane_info, 'y_pos', field(44, ctypes.c_uint32))
setattr(struct_vfio_device_gfx_plane_info, 'x_hot', field(48, ctypes.c_uint32))
setattr(struct_vfio_device_gfx_plane_info, 'y_hot', field(52, ctypes.c_uint32))
setattr(struct_vfio_device_gfx_plane_info, 'region_index', field(56, ctypes.c_uint32))
setattr(struct_vfio_device_gfx_plane_info, 'dmabuf_id', field(56, ctypes.c_uint32))
setattr(struct_vfio_device_gfx_plane_info, 'reserved', field(60, ctypes.c_uint32))
class struct_vfio_device_ioeventfd(Struct): pass
struct_vfio_device_ioeventfd.SIZE = 32
struct_vfio_device_ioeventfd._fields_ = ['argsz', 'flags', 'offset', 'data', 'fd', 'reserved']
setattr(struct_vfio_device_ioeventfd, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_device_ioeventfd, 'flags', field(4, ctypes.c_uint32))
setattr(struct_vfio_device_ioeventfd, 'offset', field(8, ctypes.c_uint64))
setattr(struct_vfio_device_ioeventfd, 'data', field(16, ctypes.c_uint64))
setattr(struct_vfio_device_ioeventfd, 'fd', field(24, ctypes.c_int32))
setattr(struct_vfio_device_ioeventfd, 'reserved', field(28, ctypes.c_uint32))
class struct_vfio_device_feature(Struct): pass
struct_vfio_device_feature.SIZE = 8
struct_vfio_device_feature._fields_ = ['argsz', 'flags', 'data']
setattr(struct_vfio_device_feature, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_device_feature, 'flags', field(4, ctypes.c_uint32))
setattr(struct_vfio_device_feature, 'data', field(8, Array(ctypes.c_ubyte, 0)))
class struct_vfio_device_bind_iommufd(Struct): pass
struct_vfio_device_bind_iommufd.SIZE = 16
struct_vfio_device_bind_iommufd._fields_ = ['argsz', 'flags', 'iommufd', 'out_devid']
setattr(struct_vfio_device_bind_iommufd, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_device_bind_iommufd, 'flags', field(4, ctypes.c_uint32))
setattr(struct_vfio_device_bind_iommufd, 'iommufd', field(8, ctypes.c_int32))
setattr(struct_vfio_device_bind_iommufd, 'out_devid', field(12, ctypes.c_uint32))
class struct_vfio_device_attach_iommufd_pt(Struct): pass
struct_vfio_device_attach_iommufd_pt.SIZE = 12
struct_vfio_device_attach_iommufd_pt._fields_ = ['argsz', 'flags', 'pt_id']
setattr(struct_vfio_device_attach_iommufd_pt, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_device_attach_iommufd_pt, 'flags', field(4, ctypes.c_uint32))
setattr(struct_vfio_device_attach_iommufd_pt, 'pt_id', field(8, ctypes.c_uint32))
class struct_vfio_device_detach_iommufd_pt(Struct): pass
struct_vfio_device_detach_iommufd_pt.SIZE = 8
struct_vfio_device_detach_iommufd_pt._fields_ = ['argsz', 'flags']
setattr(struct_vfio_device_detach_iommufd_pt, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_device_detach_iommufd_pt, 'flags', field(4, ctypes.c_uint32))
class struct_vfio_device_feature_migration(Struct): pass
struct_vfio_device_feature_migration.SIZE = 8
struct_vfio_device_feature_migration._fields_ = ['flags']
setattr(struct_vfio_device_feature_migration, 'flags', field(0, ctypes.c_uint64))
class struct_vfio_device_feature_mig_state(Struct): pass
struct_vfio_device_feature_mig_state.SIZE = 8
struct_vfio_device_feature_mig_state._fields_ = ['device_state', 'data_fd']
setattr(struct_vfio_device_feature_mig_state, 'device_state', field(0, ctypes.c_uint32))
setattr(struct_vfio_device_feature_mig_state, 'data_fd', field(4, ctypes.c_int32))
enum_vfio_device_mig_state = CEnum(ctypes.c_uint32)
VFIO_DEVICE_STATE_ERROR = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_ERROR', 0)
VFIO_DEVICE_STATE_STOP = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_STOP', 1)
VFIO_DEVICE_STATE_RUNNING = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_RUNNING', 2)
VFIO_DEVICE_STATE_STOP_COPY = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_STOP_COPY', 3)
VFIO_DEVICE_STATE_RESUMING = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_RESUMING', 4)
VFIO_DEVICE_STATE_RUNNING_P2P = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_RUNNING_P2P', 5)
VFIO_DEVICE_STATE_PRE_COPY = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_PRE_COPY', 6)
VFIO_DEVICE_STATE_PRE_COPY_P2P = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_PRE_COPY_P2P', 7)
VFIO_DEVICE_STATE_NR = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_NR', 8)

class struct_vfio_precopy_info(Struct): pass
struct_vfio_precopy_info.SIZE = 24
struct_vfio_precopy_info._fields_ = ['argsz', 'flags', 'initial_bytes', 'dirty_bytes']
setattr(struct_vfio_precopy_info, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_precopy_info, 'flags', field(4, ctypes.c_uint32))
setattr(struct_vfio_precopy_info, 'initial_bytes', field(8, ctypes.c_uint64))
setattr(struct_vfio_precopy_info, 'dirty_bytes', field(16, ctypes.c_uint64))
class struct_vfio_device_low_power_entry_with_wakeup(Struct): pass
struct_vfio_device_low_power_entry_with_wakeup.SIZE = 8
struct_vfio_device_low_power_entry_with_wakeup._fields_ = ['wakeup_eventfd', 'reserved']
setattr(struct_vfio_device_low_power_entry_with_wakeup, 'wakeup_eventfd', field(0, ctypes.c_int32))
setattr(struct_vfio_device_low_power_entry_with_wakeup, 'reserved', field(4, ctypes.c_uint32))
class struct_vfio_device_feature_dma_logging_control(Struct): pass
struct_vfio_device_feature_dma_logging_control.SIZE = 24
struct_vfio_device_feature_dma_logging_control._fields_ = ['page_size', 'num_ranges', '__reserved', 'ranges']
setattr(struct_vfio_device_feature_dma_logging_control, 'page_size', field(0, ctypes.c_uint64))
setattr(struct_vfio_device_feature_dma_logging_control, 'num_ranges', field(8, ctypes.c_uint32))
setattr(struct_vfio_device_feature_dma_logging_control, '__reserved', field(12, ctypes.c_uint32))
setattr(struct_vfio_device_feature_dma_logging_control, 'ranges', field(16, ctypes.c_uint64))
class struct_vfio_device_feature_dma_logging_range(Struct): pass
struct_vfio_device_feature_dma_logging_range.SIZE = 16
struct_vfio_device_feature_dma_logging_range._fields_ = ['iova', 'length']
setattr(struct_vfio_device_feature_dma_logging_range, 'iova', field(0, ctypes.c_uint64))
setattr(struct_vfio_device_feature_dma_logging_range, 'length', field(8, ctypes.c_uint64))
class struct_vfio_device_feature_dma_logging_report(Struct): pass
struct_vfio_device_feature_dma_logging_report.SIZE = 32
struct_vfio_device_feature_dma_logging_report._fields_ = ['iova', 'length', 'page_size', 'bitmap']
setattr(struct_vfio_device_feature_dma_logging_report, 'iova', field(0, ctypes.c_uint64))
setattr(struct_vfio_device_feature_dma_logging_report, 'length', field(8, ctypes.c_uint64))
setattr(struct_vfio_device_feature_dma_logging_report, 'page_size', field(16, ctypes.c_uint64))
setattr(struct_vfio_device_feature_dma_logging_report, 'bitmap', field(24, ctypes.c_uint64))
class struct_vfio_device_feature_mig_data_size(Struct): pass
struct_vfio_device_feature_mig_data_size.SIZE = 8
struct_vfio_device_feature_mig_data_size._fields_ = ['stop_copy_length']
setattr(struct_vfio_device_feature_mig_data_size, 'stop_copy_length', field(0, ctypes.c_uint64))
class struct_vfio_device_feature_bus_master(Struct): pass
struct_vfio_device_feature_bus_master.SIZE = 4
struct_vfio_device_feature_bus_master._fields_ = ['op']
setattr(struct_vfio_device_feature_bus_master, 'op', field(0, ctypes.c_uint32))
class struct_vfio_iommu_type1_info(Struct): pass
struct_vfio_iommu_type1_info.SIZE = 24
struct_vfio_iommu_type1_info._fields_ = ['argsz', 'flags', 'iova_pgsizes', 'cap_offset', 'pad']
setattr(struct_vfio_iommu_type1_info, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_iommu_type1_info, 'flags', field(4, ctypes.c_uint32))
setattr(struct_vfio_iommu_type1_info, 'iova_pgsizes', field(8, ctypes.c_uint64))
setattr(struct_vfio_iommu_type1_info, 'cap_offset', field(16, ctypes.c_uint32))
setattr(struct_vfio_iommu_type1_info, 'pad', field(20, ctypes.c_uint32))
class struct_vfio_iova_range(Struct): pass
struct_vfio_iova_range.SIZE = 16
struct_vfio_iova_range._fields_ = ['start', 'end']
setattr(struct_vfio_iova_range, 'start', field(0, ctypes.c_uint64))
setattr(struct_vfio_iova_range, 'end', field(8, ctypes.c_uint64))
class struct_vfio_iommu_type1_info_cap_iova_range(Struct): pass
struct_vfio_iommu_type1_info_cap_iova_range.SIZE = 16
struct_vfio_iommu_type1_info_cap_iova_range._fields_ = ['header', 'nr_iovas', 'reserved', 'iova_ranges']
setattr(struct_vfio_iommu_type1_info_cap_iova_range, 'header', field(0, struct_vfio_info_cap_header))
setattr(struct_vfio_iommu_type1_info_cap_iova_range, 'nr_iovas', field(8, ctypes.c_uint32))
setattr(struct_vfio_iommu_type1_info_cap_iova_range, 'reserved', field(12, ctypes.c_uint32))
setattr(struct_vfio_iommu_type1_info_cap_iova_range, 'iova_ranges', field(16, Array(struct_vfio_iova_range, 0)))
class struct_vfio_iommu_type1_info_cap_migration(Struct): pass
struct_vfio_iommu_type1_info_cap_migration.SIZE = 32
struct_vfio_iommu_type1_info_cap_migration._fields_ = ['header', 'flags', 'pgsize_bitmap', 'max_dirty_bitmap_size']
setattr(struct_vfio_iommu_type1_info_cap_migration, 'header', field(0, struct_vfio_info_cap_header))
setattr(struct_vfio_iommu_type1_info_cap_migration, 'flags', field(8, ctypes.c_uint32))
setattr(struct_vfio_iommu_type1_info_cap_migration, 'pgsize_bitmap', field(16, ctypes.c_uint64))
setattr(struct_vfio_iommu_type1_info_cap_migration, 'max_dirty_bitmap_size', field(24, ctypes.c_uint64))
class struct_vfio_iommu_type1_info_dma_avail(Struct): pass
struct_vfio_iommu_type1_info_dma_avail.SIZE = 12
struct_vfio_iommu_type1_info_dma_avail._fields_ = ['header', 'avail']
setattr(struct_vfio_iommu_type1_info_dma_avail, 'header', field(0, struct_vfio_info_cap_header))
setattr(struct_vfio_iommu_type1_info_dma_avail, 'avail', field(8, ctypes.c_uint32))
class struct_vfio_iommu_type1_dma_map(Struct): pass
struct_vfio_iommu_type1_dma_map.SIZE = 32
struct_vfio_iommu_type1_dma_map._fields_ = ['argsz', 'flags', 'vaddr', 'iova', 'size']
setattr(struct_vfio_iommu_type1_dma_map, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_iommu_type1_dma_map, 'flags', field(4, ctypes.c_uint32))
setattr(struct_vfio_iommu_type1_dma_map, 'vaddr', field(8, ctypes.c_uint64))
setattr(struct_vfio_iommu_type1_dma_map, 'iova', field(16, ctypes.c_uint64))
setattr(struct_vfio_iommu_type1_dma_map, 'size', field(24, ctypes.c_uint64))
class struct_vfio_bitmap(Struct): pass
struct_vfio_bitmap.SIZE = 24
struct_vfio_bitmap._fields_ = ['pgsize', 'size', 'data']
setattr(struct_vfio_bitmap, 'pgsize', field(0, ctypes.c_uint64))
setattr(struct_vfio_bitmap, 'size', field(8, ctypes.c_uint64))
setattr(struct_vfio_bitmap, 'data', field(16, Pointer(ctypes.c_uint64)))
class struct_vfio_iommu_type1_dma_unmap(Struct): pass
struct_vfio_iommu_type1_dma_unmap.SIZE = 24
struct_vfio_iommu_type1_dma_unmap._fields_ = ['argsz', 'flags', 'iova', 'size', 'data']
setattr(struct_vfio_iommu_type1_dma_unmap, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_iommu_type1_dma_unmap, 'flags', field(4, ctypes.c_uint32))
setattr(struct_vfio_iommu_type1_dma_unmap, 'iova', field(8, ctypes.c_uint64))
setattr(struct_vfio_iommu_type1_dma_unmap, 'size', field(16, ctypes.c_uint64))
setattr(struct_vfio_iommu_type1_dma_unmap, 'data', field(24, Array(ctypes.c_ubyte, 0)))
class struct_vfio_iommu_type1_dirty_bitmap(Struct): pass
struct_vfio_iommu_type1_dirty_bitmap.SIZE = 8
struct_vfio_iommu_type1_dirty_bitmap._fields_ = ['argsz', 'flags', 'data']
setattr(struct_vfio_iommu_type1_dirty_bitmap, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_iommu_type1_dirty_bitmap, 'flags', field(4, ctypes.c_uint32))
setattr(struct_vfio_iommu_type1_dirty_bitmap, 'data', field(8, Array(ctypes.c_ubyte, 0)))
class struct_vfio_iommu_type1_dirty_bitmap_get(Struct): pass
struct_vfio_iommu_type1_dirty_bitmap_get.SIZE = 40
struct_vfio_iommu_type1_dirty_bitmap_get._fields_ = ['iova', 'size', 'bitmap']
setattr(struct_vfio_iommu_type1_dirty_bitmap_get, 'iova', field(0, ctypes.c_uint64))
setattr(struct_vfio_iommu_type1_dirty_bitmap_get, 'size', field(8, ctypes.c_uint64))
setattr(struct_vfio_iommu_type1_dirty_bitmap_get, 'bitmap', field(16, struct_vfio_bitmap))
class struct_vfio_iommu_spapr_tce_ddw_info(Struct): pass
struct_vfio_iommu_spapr_tce_ddw_info.SIZE = 16
struct_vfio_iommu_spapr_tce_ddw_info._fields_ = ['pgsizes', 'max_dynamic_windows_supported', 'levels']
setattr(struct_vfio_iommu_spapr_tce_ddw_info, 'pgsizes', field(0, ctypes.c_uint64))
setattr(struct_vfio_iommu_spapr_tce_ddw_info, 'max_dynamic_windows_supported', field(8, ctypes.c_uint32))
setattr(struct_vfio_iommu_spapr_tce_ddw_info, 'levels', field(12, ctypes.c_uint32))
class struct_vfio_iommu_spapr_tce_info(Struct): pass
struct_vfio_iommu_spapr_tce_info.SIZE = 32
struct_vfio_iommu_spapr_tce_info._fields_ = ['argsz', 'flags', 'dma32_window_start', 'dma32_window_size', 'ddw']
setattr(struct_vfio_iommu_spapr_tce_info, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_iommu_spapr_tce_info, 'flags', field(4, ctypes.c_uint32))
setattr(struct_vfio_iommu_spapr_tce_info, 'dma32_window_start', field(8, ctypes.c_uint32))
setattr(struct_vfio_iommu_spapr_tce_info, 'dma32_window_size', field(12, ctypes.c_uint32))
setattr(struct_vfio_iommu_spapr_tce_info, 'ddw', field(16, struct_vfio_iommu_spapr_tce_ddw_info))
class struct_vfio_eeh_pe_err(Struct): pass
struct_vfio_eeh_pe_err.SIZE = 24
struct_vfio_eeh_pe_err._fields_ = ['type', 'func', 'addr', 'mask']
setattr(struct_vfio_eeh_pe_err, 'type', field(0, ctypes.c_uint32))
setattr(struct_vfio_eeh_pe_err, 'func', field(4, ctypes.c_uint32))
setattr(struct_vfio_eeh_pe_err, 'addr', field(8, ctypes.c_uint64))
setattr(struct_vfio_eeh_pe_err, 'mask', field(16, ctypes.c_uint64))
class struct_vfio_eeh_pe_op(Struct): pass
struct_vfio_eeh_pe_op.SIZE = 40
struct_vfio_eeh_pe_op._fields_ = ['argsz', 'flags', 'op', 'err']
setattr(struct_vfio_eeh_pe_op, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_eeh_pe_op, 'flags', field(4, ctypes.c_uint32))
setattr(struct_vfio_eeh_pe_op, 'op', field(8, ctypes.c_uint32))
setattr(struct_vfio_eeh_pe_op, 'err', field(16, struct_vfio_eeh_pe_err))
class struct_vfio_iommu_spapr_register_memory(Struct): pass
struct_vfio_iommu_spapr_register_memory.SIZE = 24
struct_vfio_iommu_spapr_register_memory._fields_ = ['argsz', 'flags', 'vaddr', 'size']
setattr(struct_vfio_iommu_spapr_register_memory, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_iommu_spapr_register_memory, 'flags', field(4, ctypes.c_uint32))
setattr(struct_vfio_iommu_spapr_register_memory, 'vaddr', field(8, ctypes.c_uint64))
setattr(struct_vfio_iommu_spapr_register_memory, 'size', field(16, ctypes.c_uint64))
class struct_vfio_iommu_spapr_tce_create(Struct): pass
struct_vfio_iommu_spapr_tce_create.SIZE = 40
struct_vfio_iommu_spapr_tce_create._fields_ = ['argsz', 'flags', 'page_shift', '__resv1', 'window_size', 'levels', '__resv2', 'start_addr']
setattr(struct_vfio_iommu_spapr_tce_create, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_iommu_spapr_tce_create, 'flags', field(4, ctypes.c_uint32))
setattr(struct_vfio_iommu_spapr_tce_create, 'page_shift', field(8, ctypes.c_uint32))
setattr(struct_vfio_iommu_spapr_tce_create, '__resv1', field(12, ctypes.c_uint32))
setattr(struct_vfio_iommu_spapr_tce_create, 'window_size', field(16, ctypes.c_uint64))
setattr(struct_vfio_iommu_spapr_tce_create, 'levels', field(24, ctypes.c_uint32))
setattr(struct_vfio_iommu_spapr_tce_create, '__resv2', field(28, ctypes.c_uint32))
setattr(struct_vfio_iommu_spapr_tce_create, 'start_addr', field(32, ctypes.c_uint64))
class struct_vfio_iommu_spapr_tce_remove(Struct): pass
struct_vfio_iommu_spapr_tce_remove.SIZE = 16
struct_vfio_iommu_spapr_tce_remove._fields_ = ['argsz', 'flags', 'start_addr']
setattr(struct_vfio_iommu_spapr_tce_remove, 'argsz', field(0, ctypes.c_uint32))
setattr(struct_vfio_iommu_spapr_tce_remove, 'flags', field(4, ctypes.c_uint32))
setattr(struct_vfio_iommu_spapr_tce_remove, 'start_addr', field(8, ctypes.c_uint64))
VFIO_API_VERSION = 0
VFIO_TYPE1_IOMMU = 1
VFIO_SPAPR_TCE_IOMMU = 2
VFIO_TYPE1v2_IOMMU = 3
VFIO_DMA_CC_IOMMU = 4
VFIO_EEH = 5
VFIO_TYPE1_NESTING_IOMMU = 6
VFIO_SPAPR_TCE_v2_IOMMU = 7
VFIO_NOIOMMU_IOMMU = 8
VFIO_UNMAP_ALL = 9
VFIO_UPDATE_VADDR = 10
VFIO_TYPE = (';')
VFIO_BASE = 100
VFIO_GET_API_VERSION = _IO(VFIO_TYPE, VFIO_BASE + 0)
VFIO_CHECK_EXTENSION = _IO(VFIO_TYPE, VFIO_BASE + 1)
VFIO_SET_IOMMU = _IO(VFIO_TYPE, VFIO_BASE + 2)
VFIO_GROUP_FLAGS_VIABLE = (1 << 0)
VFIO_GROUP_FLAGS_CONTAINER_SET = (1 << 1)
VFIO_GROUP_GET_STATUS = _IO(VFIO_TYPE, VFIO_BASE + 3)
VFIO_GROUP_SET_CONTAINER = _IO(VFIO_TYPE, VFIO_BASE + 4)
VFIO_GROUP_UNSET_CONTAINER = _IO(VFIO_TYPE, VFIO_BASE + 5)
VFIO_GROUP_GET_DEVICE_FD = _IO(VFIO_TYPE, VFIO_BASE + 6)
VFIO_DEVICE_FLAGS_RESET = (1 << 0)
VFIO_DEVICE_FLAGS_PCI = (1 << 1)
VFIO_DEVICE_FLAGS_PLATFORM = (1 << 2)
VFIO_DEVICE_FLAGS_AMBA = (1 << 3)
VFIO_DEVICE_FLAGS_CCW = (1 << 4)
VFIO_DEVICE_FLAGS_AP = (1 << 5)
VFIO_DEVICE_FLAGS_FSL_MC = (1 << 6)
VFIO_DEVICE_FLAGS_CAPS = (1 << 7)
VFIO_DEVICE_FLAGS_CDX = (1 << 8)
VFIO_DEVICE_GET_INFO = _IO(VFIO_TYPE, VFIO_BASE + 7)
VFIO_DEVICE_API_PCI_STRING = "vfio-pci"
VFIO_DEVICE_API_PLATFORM_STRING = "vfio-platform"
VFIO_DEVICE_API_AMBA_STRING = "vfio-amba"
VFIO_DEVICE_API_CCW_STRING = "vfio-ccw"
VFIO_DEVICE_API_AP_STRING = "vfio-ap"
VFIO_DEVICE_INFO_CAP_ZPCI_BASE = 1
VFIO_DEVICE_INFO_CAP_ZPCI_GROUP = 2
VFIO_DEVICE_INFO_CAP_ZPCI_UTIL = 3
VFIO_DEVICE_INFO_CAP_ZPCI_PFIP = 4
VFIO_DEVICE_INFO_CAP_PCI_ATOMIC_COMP = 5
VFIO_PCI_ATOMIC_COMP32 = (1 << 0)
VFIO_PCI_ATOMIC_COMP64 = (1 << 1)
VFIO_PCI_ATOMIC_COMP128 = (1 << 2)
VFIO_REGION_INFO_FLAG_READ = (1 << 0)
VFIO_REGION_INFO_FLAG_WRITE = (1 << 1)
VFIO_REGION_INFO_FLAG_MMAP = (1 << 2)
VFIO_REGION_INFO_FLAG_CAPS = (1 << 3)
VFIO_DEVICE_GET_REGION_INFO = _IO(VFIO_TYPE, VFIO_BASE + 8)
VFIO_REGION_INFO_CAP_SPARSE_MMAP = 1
VFIO_REGION_INFO_CAP_TYPE = 2
VFIO_REGION_TYPE_PCI_VENDOR_TYPE = (1 << 31)
VFIO_REGION_TYPE_PCI_VENDOR_MASK = (0xffff)
VFIO_REGION_TYPE_GFX = (1)
VFIO_REGION_TYPE_CCW = (2)
VFIO_REGION_TYPE_MIGRATION_DEPRECATED = (3)
VFIO_REGION_SUBTYPE_INTEL_IGD_OPREGION = (1)
VFIO_REGION_SUBTYPE_INTEL_IGD_HOST_CFG = (2)
VFIO_REGION_SUBTYPE_INTEL_IGD_LPC_CFG = (3)
VFIO_REGION_SUBTYPE_NVIDIA_NVLINK2_RAM = (1)
VFIO_REGION_SUBTYPE_IBM_NVLINK2_ATSD = (1)
VFIO_REGION_SUBTYPE_GFX_EDID = (1)
VFIO_DEVICE_GFX_LINK_STATE_UP = 1
VFIO_DEVICE_GFX_LINK_STATE_DOWN = 2
VFIO_REGION_SUBTYPE_CCW_ASYNC_CMD = (1)
VFIO_REGION_SUBTYPE_CCW_SCHIB = (2)
VFIO_REGION_SUBTYPE_CCW_CRW = (3)
VFIO_REGION_SUBTYPE_MIGRATION_DEPRECATED = (1)
VFIO_DEVICE_STATE_V1_STOP = (0)
VFIO_DEVICE_STATE_V1_RUNNING = (1 << 0)
VFIO_DEVICE_STATE_V1_SAVING = (1 << 1)
VFIO_DEVICE_STATE_V1_RESUMING = (1 << 2)
VFIO_DEVICE_STATE_MASK = (VFIO_DEVICE_STATE_V1_RUNNING | VFIO_DEVICE_STATE_V1_SAVING | VFIO_DEVICE_STATE_V1_RESUMING)
VFIO_DEVICE_STATE_IS_ERROR = lambda state: ((state & VFIO_DEVICE_STATE_MASK) == (VFIO_DEVICE_STATE_V1_SAVING | VFIO_DEVICE_STATE_V1_RESUMING))
VFIO_DEVICE_STATE_SET_ERROR = lambda state: ((state & ~VFIO_DEVICE_STATE_MASK) | VFIO_DEVICE_STATE_V1_SAVING | VFIO_DEVICE_STATE_V1_RESUMING)
VFIO_REGION_INFO_CAP_MSIX_MAPPABLE = 3
VFIO_REGION_INFO_CAP_NVLINK2_SSATGT = 4
VFIO_REGION_INFO_CAP_NVLINK2_LNKSPD = 5
VFIO_IRQ_INFO_EVENTFD = (1 << 0)
VFIO_IRQ_INFO_MASKABLE = (1 << 1)
VFIO_IRQ_INFO_AUTOMASKED = (1 << 2)
VFIO_IRQ_INFO_NORESIZE = (1 << 3)
VFIO_DEVICE_GET_IRQ_INFO = _IO(VFIO_TYPE, VFIO_BASE + 9)
VFIO_IRQ_SET_DATA_NONE = (1 << 0)
VFIO_IRQ_SET_DATA_BOOL = (1 << 1)
VFIO_IRQ_SET_DATA_EVENTFD = (1 << 2)
VFIO_IRQ_SET_ACTION_MASK = (1 << 3)
VFIO_IRQ_SET_ACTION_UNMASK = (1 << 4)
VFIO_IRQ_SET_ACTION_TRIGGER = (1 << 5)
VFIO_DEVICE_SET_IRQS = _IO(VFIO_TYPE, VFIO_BASE + 10)
VFIO_IRQ_SET_DATA_TYPE_MASK = (VFIO_IRQ_SET_DATA_NONE | VFIO_IRQ_SET_DATA_BOOL | VFIO_IRQ_SET_DATA_EVENTFD)
VFIO_IRQ_SET_ACTION_TYPE_MASK = (VFIO_IRQ_SET_ACTION_MASK | VFIO_IRQ_SET_ACTION_UNMASK | VFIO_IRQ_SET_ACTION_TRIGGER)
VFIO_DEVICE_RESET = _IO(VFIO_TYPE, VFIO_BASE + 11)
VFIO_PCI_DEVID_OWNED = 0
VFIO_PCI_DEVID_NOT_OWNED = -1
VFIO_PCI_HOT_RESET_FLAG_DEV_ID = (1 << 0)
VFIO_PCI_HOT_RESET_FLAG_DEV_ID_OWNED = (1 << 1)
VFIO_DEVICE_GET_PCI_HOT_RESET_INFO = _IO(VFIO_TYPE, VFIO_BASE + 12)
VFIO_DEVICE_PCI_HOT_RESET = _IO(VFIO_TYPE, VFIO_BASE + 13)
VFIO_GFX_PLANE_TYPE_PROBE = (1 << 0)
VFIO_GFX_PLANE_TYPE_DMABUF = (1 << 1)
VFIO_GFX_PLANE_TYPE_REGION = (1 << 2)
VFIO_DEVICE_QUERY_GFX_PLANE = _IO(VFIO_TYPE, VFIO_BASE + 14)
VFIO_DEVICE_GET_GFX_DMABUF = _IO(VFIO_TYPE, VFIO_BASE + 15)
VFIO_DEVICE_IOEVENTFD_8 = (1 << 0)
VFIO_DEVICE_IOEVENTFD_16 = (1 << 1)
VFIO_DEVICE_IOEVENTFD_32 = (1 << 2)
VFIO_DEVICE_IOEVENTFD_64 = (1 << 3)
VFIO_DEVICE_IOEVENTFD_SIZE_MASK = (0xf)
VFIO_DEVICE_IOEVENTFD = _IO(VFIO_TYPE, VFIO_BASE + 16)
VFIO_DEVICE_FEATURE_MASK = (0xffff)
VFIO_DEVICE_FEATURE_GET = (1 << 16)
VFIO_DEVICE_FEATURE_SET = (1 << 17)
VFIO_DEVICE_FEATURE_PROBE = (1 << 18)
VFIO_DEVICE_FEATURE = _IO(VFIO_TYPE, VFIO_BASE + 17)
VFIO_DEVICE_BIND_IOMMUFD = _IO(VFIO_TYPE, VFIO_BASE + 18)
VFIO_DEVICE_ATTACH_IOMMUFD_PT = _IO(VFIO_TYPE, VFIO_BASE + 19)
VFIO_DEVICE_DETACH_IOMMUFD_PT = _IO(VFIO_TYPE, VFIO_BASE + 20)
VFIO_DEVICE_FEATURE_PCI_VF_TOKEN = (0)
VFIO_MIGRATION_STOP_COPY = (1 << 0)
VFIO_MIGRATION_P2P = (1 << 1)
VFIO_MIGRATION_PRE_COPY = (1 << 2)
VFIO_DEVICE_FEATURE_MIGRATION = 1
VFIO_DEVICE_FEATURE_MIG_DEVICE_STATE = 2
VFIO_MIG_GET_PRECOPY_INFO = _IO(VFIO_TYPE, VFIO_BASE + 21)
VFIO_DEVICE_FEATURE_LOW_POWER_ENTRY = 3
VFIO_DEVICE_FEATURE_LOW_POWER_ENTRY_WITH_WAKEUP = 4
VFIO_DEVICE_FEATURE_LOW_POWER_EXIT = 5
VFIO_DEVICE_FEATURE_DMA_LOGGING_START = 6
VFIO_DEVICE_FEATURE_DMA_LOGGING_STOP = 7
VFIO_DEVICE_FEATURE_DMA_LOGGING_REPORT = 8
VFIO_DEVICE_FEATURE_MIG_DATA_SIZE = 9
VFIO_DEVICE_FEATURE_CLEAR_MASTER = 0
VFIO_DEVICE_FEATURE_SET_MASTER = 1
VFIO_DEVICE_FEATURE_BUS_MASTER = 10
VFIO_IOMMU_INFO_PGSIZES = (1 << 0)
VFIO_IOMMU_INFO_CAPS = (1 << 1)
VFIO_IOMMU_TYPE1_INFO_CAP_IOVA_RANGE = 1
VFIO_IOMMU_TYPE1_INFO_CAP_MIGRATION = 2
VFIO_IOMMU_TYPE1_INFO_DMA_AVAIL = 3
VFIO_IOMMU_GET_INFO = _IO(VFIO_TYPE, VFIO_BASE + 12)
VFIO_DMA_MAP_FLAG_READ = (1 << 0)
VFIO_DMA_MAP_FLAG_WRITE = (1 << 1)
VFIO_DMA_MAP_FLAG_VADDR = (1 << 2)
VFIO_IOMMU_MAP_DMA = _IO(VFIO_TYPE, VFIO_BASE + 13)
VFIO_DMA_UNMAP_FLAG_GET_DIRTY_BITMAP = (1 << 0)
VFIO_DMA_UNMAP_FLAG_ALL = (1 << 1)
VFIO_DMA_UNMAP_FLAG_VADDR = (1 << 2)
VFIO_IOMMU_UNMAP_DMA = _IO(VFIO_TYPE, VFIO_BASE + 14)
VFIO_IOMMU_ENABLE = _IO(VFIO_TYPE, VFIO_BASE + 15)
VFIO_IOMMU_DISABLE = _IO(VFIO_TYPE, VFIO_BASE + 16)
VFIO_IOMMU_DIRTY_PAGES_FLAG_START = (1 << 0)
VFIO_IOMMU_DIRTY_PAGES_FLAG_STOP = (1 << 1)
VFIO_IOMMU_DIRTY_PAGES_FLAG_GET_BITMAP = (1 << 2)
VFIO_IOMMU_DIRTY_PAGES = _IO(VFIO_TYPE, VFIO_BASE + 17)
VFIO_IOMMU_SPAPR_INFO_DDW = (1 << 0)
VFIO_IOMMU_SPAPR_TCE_GET_INFO = _IO(VFIO_TYPE, VFIO_BASE + 12)
VFIO_EEH_PE_DISABLE = 0
VFIO_EEH_PE_ENABLE = 1
VFIO_EEH_PE_UNFREEZE_IO = 2
VFIO_EEH_PE_UNFREEZE_DMA = 3
VFIO_EEH_PE_GET_STATE = 4
VFIO_EEH_PE_STATE_NORMAL = 0
VFIO_EEH_PE_STATE_RESET = 1
VFIO_EEH_PE_STATE_STOPPED = 2
VFIO_EEH_PE_STATE_STOPPED_DMA = 4
VFIO_EEH_PE_STATE_UNAVAIL = 5
VFIO_EEH_PE_RESET_DEACTIVATE = 5
VFIO_EEH_PE_RESET_HOT = 6
VFIO_EEH_PE_RESET_FUNDAMENTAL = 7
VFIO_EEH_PE_CONFIGURE = 8
VFIO_EEH_PE_INJECT_ERR = 9
VFIO_EEH_PE_OP = _IO(VFIO_TYPE, VFIO_BASE + 21)
VFIO_IOMMU_SPAPR_REGISTER_MEMORY = _IO(VFIO_TYPE, VFIO_BASE + 17)
VFIO_IOMMU_SPAPR_UNREGISTER_MEMORY = _IO(VFIO_TYPE, VFIO_BASE + 18)
VFIO_IOMMU_SPAPR_TCE_CREATE = _IO(VFIO_TYPE, VFIO_BASE + 19)
VFIO_IOMMU_SPAPR_TCE_REMOVE = _IO(VFIO_TYPE, VFIO_BASE + 20)