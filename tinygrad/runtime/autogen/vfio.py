# mypy: ignore-errors
import ctypes
from tinygrad.helpers import CEnum, _IO, _IOW, _IOR, _IOWR
class struct_vfio_info_cap_header(ctypes.Structure): pass
__u16 = ctypes.c_ushort
__u32 = ctypes.c_uint
struct_vfio_info_cap_header._fields_ = [
  ('id', ctypes.c_ushort),
  ('version', ctypes.c_ushort),
  ('next', ctypes.c_uint),
]
class struct_vfio_group_status(ctypes.Structure): pass
struct_vfio_group_status._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
]
class struct_vfio_device_info(ctypes.Structure): pass
struct_vfio_device_info._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('num_regions', ctypes.c_uint),
  ('num_irqs', ctypes.c_uint),
  ('cap_offset', ctypes.c_uint),
  ('pad', ctypes.c_uint),
]
class struct_vfio_device_info_cap_pci_atomic_comp(ctypes.Structure): pass
struct_vfio_device_info_cap_pci_atomic_comp._fields_ = [
  ('header', struct_vfio_info_cap_header),
  ('flags', ctypes.c_uint),
  ('reserved', ctypes.c_uint),
]
class struct_vfio_region_info(ctypes.Structure): pass
__u64 = ctypes.c_ulonglong
struct_vfio_region_info._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('index', ctypes.c_uint),
  ('cap_offset', ctypes.c_uint),
  ('size', ctypes.c_ulonglong),
  ('offset', ctypes.c_ulonglong),
]
class struct_vfio_region_sparse_mmap_area(ctypes.Structure): pass
struct_vfio_region_sparse_mmap_area._fields_ = [
  ('offset', ctypes.c_ulonglong),
  ('size', ctypes.c_ulonglong),
]
class struct_vfio_region_info_cap_sparse_mmap(ctypes.Structure): pass
struct_vfio_region_info_cap_sparse_mmap._fields_ = [
  ('header', struct_vfio_info_cap_header),
  ('nr_areas', ctypes.c_uint),
  ('reserved', ctypes.c_uint),
  ('areas', (struct_vfio_region_sparse_mmap_area * 0)),
]
class struct_vfio_region_info_cap_type(ctypes.Structure): pass
struct_vfio_region_info_cap_type._fields_ = [
  ('header', struct_vfio_info_cap_header),
  ('type', ctypes.c_uint),
  ('subtype', ctypes.c_uint),
]
class struct_vfio_region_gfx_edid(ctypes.Structure): pass
struct_vfio_region_gfx_edid._fields_ = [
  ('edid_offset', ctypes.c_uint),
  ('edid_max_size', ctypes.c_uint),
  ('edid_size', ctypes.c_uint),
  ('max_xres', ctypes.c_uint),
  ('max_yres', ctypes.c_uint),
  ('link_state', ctypes.c_uint),
]
class struct_vfio_device_migration_info(ctypes.Structure): pass
struct_vfio_device_migration_info._fields_ = [
  ('device_state', ctypes.c_uint),
  ('reserved', ctypes.c_uint),
  ('pending_bytes', ctypes.c_ulonglong),
  ('data_offset', ctypes.c_ulonglong),
  ('data_size', ctypes.c_ulonglong),
]
class struct_vfio_region_info_cap_nvlink2_ssatgt(ctypes.Structure): pass
struct_vfio_region_info_cap_nvlink2_ssatgt._fields_ = [
  ('header', struct_vfio_info_cap_header),
  ('tgt', ctypes.c_ulonglong),
]
class struct_vfio_region_info_cap_nvlink2_lnkspd(ctypes.Structure): pass
struct_vfio_region_info_cap_nvlink2_lnkspd._fields_ = [
  ('header', struct_vfio_info_cap_header),
  ('link_speed', ctypes.c_uint),
  ('__pad', ctypes.c_uint),
]
class struct_vfio_irq_info(ctypes.Structure): pass
struct_vfio_irq_info._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('index', ctypes.c_uint),
  ('count', ctypes.c_uint),
]
class struct_vfio_irq_set(ctypes.Structure): pass
__u8 = ctypes.c_ubyte
struct_vfio_irq_set._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('index', ctypes.c_uint),
  ('start', ctypes.c_uint),
  ('count', ctypes.c_uint),
  ('data', (ctypes.c_ubyte * 0)),
]
_anonenum0 = CEnum(ctypes.c_uint)
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

_anonenum1 = CEnum(ctypes.c_uint)
VFIO_PCI_INTX_IRQ_INDEX = _anonenum1.define('VFIO_PCI_INTX_IRQ_INDEX', 0)
VFIO_PCI_MSI_IRQ_INDEX = _anonenum1.define('VFIO_PCI_MSI_IRQ_INDEX', 1)
VFIO_PCI_MSIX_IRQ_INDEX = _anonenum1.define('VFIO_PCI_MSIX_IRQ_INDEX', 2)
VFIO_PCI_ERR_IRQ_INDEX = _anonenum1.define('VFIO_PCI_ERR_IRQ_INDEX', 3)
VFIO_PCI_REQ_IRQ_INDEX = _anonenum1.define('VFIO_PCI_REQ_IRQ_INDEX', 4)
VFIO_PCI_NUM_IRQS = _anonenum1.define('VFIO_PCI_NUM_IRQS', 5)

_anonenum2 = CEnum(ctypes.c_uint)
VFIO_CCW_CONFIG_REGION_INDEX = _anonenum2.define('VFIO_CCW_CONFIG_REGION_INDEX', 0)
VFIO_CCW_NUM_REGIONS = _anonenum2.define('VFIO_CCW_NUM_REGIONS', 1)

_anonenum3 = CEnum(ctypes.c_uint)
VFIO_CCW_IO_IRQ_INDEX = _anonenum3.define('VFIO_CCW_IO_IRQ_INDEX', 0)
VFIO_CCW_CRW_IRQ_INDEX = _anonenum3.define('VFIO_CCW_CRW_IRQ_INDEX', 1)
VFIO_CCW_REQ_IRQ_INDEX = _anonenum3.define('VFIO_CCW_REQ_IRQ_INDEX', 2)
VFIO_CCW_NUM_IRQS = _anonenum3.define('VFIO_CCW_NUM_IRQS', 3)

_anonenum4 = CEnum(ctypes.c_uint)
VFIO_AP_REQ_IRQ_INDEX = _anonenum4.define('VFIO_AP_REQ_IRQ_INDEX', 0)
VFIO_AP_NUM_IRQS = _anonenum4.define('VFIO_AP_NUM_IRQS', 1)

class struct_vfio_pci_dependent_device(ctypes.Structure): pass
class _anonunion5(ctypes.Union): pass
_anonunion5._fields_ = [
  ('group_id', ctypes.c_uint),
  ('devid', ctypes.c_uint),
]
struct_vfio_pci_dependent_device._anonymous_ = ['_0']
struct_vfio_pci_dependent_device._fields_ = [
  ('_0', _anonunion5),
  ('segment', ctypes.c_ushort),
  ('bus', ctypes.c_ubyte),
  ('devfn', ctypes.c_ubyte),
]
class struct_vfio_pci_hot_reset_info(ctypes.Structure): pass
struct_vfio_pci_hot_reset_info._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('count', ctypes.c_uint),
  ('devices', (struct_vfio_pci_dependent_device * 0)),
]
class struct_vfio_pci_hot_reset(ctypes.Structure): pass
__s32 = ctypes.c_int
struct_vfio_pci_hot_reset._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('count', ctypes.c_uint),
  ('group_fds', (ctypes.c_int * 0)),
]
class struct_vfio_device_gfx_plane_info(ctypes.Structure): pass
class _anonunion6(ctypes.Union): pass
_anonunion6._fields_ = [
  ('region_index', ctypes.c_uint),
  ('dmabuf_id', ctypes.c_uint),
]
struct_vfio_device_gfx_plane_info._anonymous_ = ['_0']
struct_vfio_device_gfx_plane_info._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('drm_plane_type', ctypes.c_uint),
  ('drm_format', ctypes.c_uint),
  ('drm_format_mod', ctypes.c_ulonglong),
  ('width', ctypes.c_uint),
  ('height', ctypes.c_uint),
  ('stride', ctypes.c_uint),
  ('size', ctypes.c_uint),
  ('x_pos', ctypes.c_uint),
  ('y_pos', ctypes.c_uint),
  ('x_hot', ctypes.c_uint),
  ('y_hot', ctypes.c_uint),
  ('_0', _anonunion6),
  ('reserved', ctypes.c_uint),
]
class struct_vfio_device_ioeventfd(ctypes.Structure): pass
struct_vfio_device_ioeventfd._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('offset', ctypes.c_ulonglong),
  ('data', ctypes.c_ulonglong),
  ('fd', ctypes.c_int),
  ('reserved', ctypes.c_uint),
]
class struct_vfio_device_feature(ctypes.Structure): pass
struct_vfio_device_feature._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('data', (ctypes.c_ubyte * 0)),
]
class struct_vfio_device_bind_iommufd(ctypes.Structure): pass
struct_vfio_device_bind_iommufd._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('iommufd', ctypes.c_int),
  ('out_devid', ctypes.c_uint),
]
class struct_vfio_device_attach_iommufd_pt(ctypes.Structure): pass
struct_vfio_device_attach_iommufd_pt._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('pt_id', ctypes.c_uint),
]
class struct_vfio_device_detach_iommufd_pt(ctypes.Structure): pass
struct_vfio_device_detach_iommufd_pt._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
]
class struct_vfio_device_feature_migration(ctypes.Structure): pass
struct_vfio_device_feature_migration._fields_ = [
  ('flags', ctypes.c_ulonglong),
]
class struct_vfio_device_feature_mig_state(ctypes.Structure): pass
struct_vfio_device_feature_mig_state._fields_ = [
  ('device_state', ctypes.c_uint),
  ('data_fd', ctypes.c_int),
]
enum_vfio_device_mig_state = CEnum(ctypes.c_uint)
VFIO_DEVICE_STATE_ERROR = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_ERROR', 0)
VFIO_DEVICE_STATE_STOP = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_STOP', 1)
VFIO_DEVICE_STATE_RUNNING = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_RUNNING', 2)
VFIO_DEVICE_STATE_STOP_COPY = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_STOP_COPY', 3)
VFIO_DEVICE_STATE_RESUMING = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_RESUMING', 4)
VFIO_DEVICE_STATE_RUNNING_P2P = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_RUNNING_P2P', 5)
VFIO_DEVICE_STATE_PRE_COPY = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_PRE_COPY', 6)
VFIO_DEVICE_STATE_PRE_COPY_P2P = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_PRE_COPY_P2P', 7)
VFIO_DEVICE_STATE_NR = enum_vfio_device_mig_state.define('VFIO_DEVICE_STATE_NR', 8)

class struct_vfio_precopy_info(ctypes.Structure): pass
struct_vfio_precopy_info._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('initial_bytes', ctypes.c_ulonglong),
  ('dirty_bytes', ctypes.c_ulonglong),
]
class struct_vfio_device_low_power_entry_with_wakeup(ctypes.Structure): pass
struct_vfio_device_low_power_entry_with_wakeup._fields_ = [
  ('wakeup_eventfd', ctypes.c_int),
  ('reserved', ctypes.c_uint),
]
class struct_vfio_device_feature_dma_logging_control(ctypes.Structure): pass
struct_vfio_device_feature_dma_logging_control._fields_ = [
  ('page_size', ctypes.c_ulonglong),
  ('num_ranges', ctypes.c_uint),
  ('__reserved', ctypes.c_uint),
  ('ranges', ctypes.c_ulonglong),
]
class struct_vfio_device_feature_dma_logging_range(ctypes.Structure): pass
struct_vfio_device_feature_dma_logging_range._fields_ = [
  ('iova', ctypes.c_ulonglong),
  ('length', ctypes.c_ulonglong),
]
class struct_vfio_device_feature_dma_logging_report(ctypes.Structure): pass
struct_vfio_device_feature_dma_logging_report._fields_ = [
  ('iova', ctypes.c_ulonglong),
  ('length', ctypes.c_ulonglong),
  ('page_size', ctypes.c_ulonglong),
  ('bitmap', ctypes.c_ulonglong),
]
class struct_vfio_device_feature_mig_data_size(ctypes.Structure): pass
struct_vfio_device_feature_mig_data_size._fields_ = [
  ('stop_copy_length', ctypes.c_ulonglong),
]
class struct_vfio_device_feature_bus_master(ctypes.Structure): pass
struct_vfio_device_feature_bus_master._fields_ = [
  ('op', ctypes.c_uint),
]
class struct_vfio_iommu_type1_info(ctypes.Structure): pass
struct_vfio_iommu_type1_info._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('iova_pgsizes', ctypes.c_ulonglong),
  ('cap_offset', ctypes.c_uint),
  ('pad', ctypes.c_uint),
]
class struct_vfio_iova_range(ctypes.Structure): pass
struct_vfio_iova_range._fields_ = [
  ('start', ctypes.c_ulonglong),
  ('end', ctypes.c_ulonglong),
]
class struct_vfio_iommu_type1_info_cap_iova_range(ctypes.Structure): pass
struct_vfio_iommu_type1_info_cap_iova_range._fields_ = [
  ('header', struct_vfio_info_cap_header),
  ('nr_iovas', ctypes.c_uint),
  ('reserved', ctypes.c_uint),
  ('iova_ranges', (struct_vfio_iova_range * 0)),
]
class struct_vfio_iommu_type1_info_cap_migration(ctypes.Structure): pass
struct_vfio_iommu_type1_info_cap_migration._fields_ = [
  ('header', struct_vfio_info_cap_header),
  ('flags', ctypes.c_uint),
  ('pgsize_bitmap', ctypes.c_ulonglong),
  ('max_dirty_bitmap_size', ctypes.c_ulonglong),
]
class struct_vfio_iommu_type1_info_dma_avail(ctypes.Structure): pass
struct_vfio_iommu_type1_info_dma_avail._fields_ = [
  ('header', struct_vfio_info_cap_header),
  ('avail', ctypes.c_uint),
]
class struct_vfio_iommu_type1_dma_map(ctypes.Structure): pass
struct_vfio_iommu_type1_dma_map._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('vaddr', ctypes.c_ulonglong),
  ('iova', ctypes.c_ulonglong),
  ('size', ctypes.c_ulonglong),
]
class struct_vfio_bitmap(ctypes.Structure): pass
struct_vfio_bitmap._fields_ = [
  ('pgsize', ctypes.c_ulonglong),
  ('size', ctypes.c_ulonglong),
  ('data', ctypes.POINTER(ctypes.c_ulonglong)),
]
class struct_vfio_iommu_type1_dma_unmap(ctypes.Structure): pass
struct_vfio_iommu_type1_dma_unmap._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('iova', ctypes.c_ulonglong),
  ('size', ctypes.c_ulonglong),
  ('data', (ctypes.c_ubyte * 0)),
]
class struct_vfio_iommu_type1_dirty_bitmap(ctypes.Structure): pass
struct_vfio_iommu_type1_dirty_bitmap._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('data', (ctypes.c_ubyte * 0)),
]
class struct_vfio_iommu_type1_dirty_bitmap_get(ctypes.Structure): pass
struct_vfio_iommu_type1_dirty_bitmap_get._fields_ = [
  ('iova', ctypes.c_ulonglong),
  ('size', ctypes.c_ulonglong),
  ('bitmap', struct_vfio_bitmap),
]
class struct_vfio_iommu_spapr_tce_ddw_info(ctypes.Structure): pass
struct_vfio_iommu_spapr_tce_ddw_info._fields_ = [
  ('pgsizes', ctypes.c_ulonglong),
  ('max_dynamic_windows_supported', ctypes.c_uint),
  ('levels', ctypes.c_uint),
]
class struct_vfio_iommu_spapr_tce_info(ctypes.Structure): pass
struct_vfio_iommu_spapr_tce_info._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('dma32_window_start', ctypes.c_uint),
  ('dma32_window_size', ctypes.c_uint),
  ('ddw', struct_vfio_iommu_spapr_tce_ddw_info),
]
class struct_vfio_eeh_pe_err(ctypes.Structure): pass
struct_vfio_eeh_pe_err._fields_ = [
  ('type', ctypes.c_uint),
  ('func', ctypes.c_uint),
  ('addr', ctypes.c_ulonglong),
  ('mask', ctypes.c_ulonglong),
]
class struct_vfio_eeh_pe_op(ctypes.Structure): pass
class _anonunion7(ctypes.Union): pass
_anonunion7._fields_ = [
  ('err', struct_vfio_eeh_pe_err),
]
struct_vfio_eeh_pe_op._anonymous_ = ['_0']
struct_vfio_eeh_pe_op._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('op', ctypes.c_uint),
  ('_0', _anonunion7),
]
class struct_vfio_iommu_spapr_register_memory(ctypes.Structure): pass
struct_vfio_iommu_spapr_register_memory._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('vaddr', ctypes.c_ulonglong),
  ('size', ctypes.c_ulonglong),
]
class struct_vfio_iommu_spapr_tce_create(ctypes.Structure): pass
struct_vfio_iommu_spapr_tce_create._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('page_shift', ctypes.c_uint),
  ('__resv1', ctypes.c_uint),
  ('window_size', ctypes.c_ulonglong),
  ('levels', ctypes.c_uint),
  ('__resv2', ctypes.c_uint),
  ('start_addr', ctypes.c_ulonglong),
]
class struct_vfio_iommu_spapr_tce_remove(ctypes.Structure): pass
struct_vfio_iommu_spapr_tce_remove._fields_ = [
  ('argsz', ctypes.c_uint),
  ('flags', ctypes.c_uint),
  ('start_addr', ctypes.c_ulonglong),
]
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