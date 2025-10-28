import importlib, pathlib, subprocess

def sys(cmd): return subprocess.check_output(cmd.split()).decode().strip()

root = (here:=pathlib.Path(__file__).parent).parent.parent.parent
nv_src = "https://github.com/NVIDIA/open-gpu-kernel-modules/archive/81fe4fb417c8ac3b9bdcc1d56827d116743892a5.tar.gz"

def load(name, *args, **kwargs):
  if not (f:=(here / f"{name}.py")).exists(): f.write_text(importlib.import_module(f"{__name__}.autogen").gen(*args, **kwargs))
  return importlib.import_module(f"{__name__}.{name}")

def __getattr__(nm):
  match nm:
    case "libc": return load("libc", "ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)", lambda: (
  [i for i in sys("dpkg -L libc6-dev").split() if 'sys/mman.h' in i or 'sys/syscall.h' in i] +
  ["/usr/include/string.h", "/usr/include/elf.h", "/usr/include/unistd.h", "/usr/include/asm-generic/mman-common.h"]))
    case "opencl": return load("opencl", "ctypes.CDLL(ctypes.util.find_library('OpenCL'))", ["/usr/include/CL/cl.h"])
    case "cuda": return load("cuda", "ctypes.CDLL(ctypes.util.find_library('cuda'))", ["/usr/include/cuda.h"], args=["-D__CUDA_API_VERSION_INTERNAL"])
    case "nvrtc": return load("nvrtc", "ctypes.CDLL(ctypes.util.find_library('nvrtc'))", ["/usr/local/cuda/include/nvrtc.h"])
    case "nvjitlink": load("nvjitlink", "ctypes.CDLL(ctypes.util.find_library('nvJitLink'))", ["/usr/local/cuda/include/nvJitLink.h"])
    case "kfd": return load("kfd", None, ["/usr/include/linux/kfd_ioctl.h"])
    case "nv_gpu":
      return load("nv_gpu",None,[*[root/"extra/nv_gpu_driver"/s for s in ["clc6c0qmd.h","clcec0qmd.h"]],"{}/kernel-open/common/inc/nvmisc.h",*["{}"+
  f"/src/common/sdk/nvidia/inc/class/cl{s}.h" for s in ["0000","0080","2080","2080_notification","c56f","c86f","c96f","c761","83de","c6c0","cdc0"]],
  *[f"{{}}/kernel-open/nvidia-uvm/{s}.h" for s in ["clc6b5","clc9b5","uvm_ioctl","uvm_linux_ioctl","hwref/ampere/ga100/dev_fault"]],
  *[f"{{}}/src/nvidia/arch/nvalloc/unix/include/nv{s}.h" for s in ["_escape","-ioctl","-ioctl-numbers","-ioctl-numa","-unix-nvos-params-wrappers"]],
  *[f"{{}}/src/common/sdk/nvidia/inc/{s}.h" for s in ["alloc/alloc_channel","nvos","ctrl/ctrlc36f","ctrl/ctrlcb33","ctrl/ctrla06c","ctrl/ctrl90f1"]],
  *[f"{{}}/src/common/sdk/nvidia/inc/ctrl/ctrl{s}/*.h" for s in ["0000","0080","2080","83de"]],
  "{}/kernel-open/common/inc/nvstatus.h","{}/src/nvidia/generated/g_allclasses.h"],
  ["-I{}/src/common/inc", "-I{}/kernel-open/nvidia-uvm","-I{}/kernel-open/common/inc","-I{}/src/common/sdk/nvidia/inc",
   "-I{}/src/nvidia/arch/nvalloc/unix/include","-I{}/src/common/sdk/nvidia/inc/ctrl"], rules=[(r'MW\(([^:]+):(.+)\)',r'(\1, \2)')], tarball=nv_src)
    case "nv": return load("nv",None,[f"{{}}/src/nvidia/inc/kernel/gpu/{s}.h" for s in ["fsp/kern_fsp_cot_payload","gsp/gsp_init_args"]]+["{}/src/"+
  f"nvidia/arch/nvalloc/common/inc/{s}.h" for s in ["gsp/gspifpub","gsp/gsp_fw_wpr_meta","gsp/gsp_fw_sr_meta","rmRiscvUcode","fsp/fsp_nvdm_format"]]+
  [f"{{}}/src/nvidia/inc/kernel/vgpu/{s}.h" for s in ["rpc_headers","rpc_global_enums"]]+["{}/src/common/uproc/os/common/include/libos_init_args.h",
  "{}/src/common/shared/msgq/inc/msgq/msgq_priv.h","{}/src/nvidia/generated/g_rpc-structures.h",root/"extra/nv_gpu_driver/g_rpc-message-header.h",
  root/"extra/nv_gpu_driver/gsp_static_config.h",root/"extra/nv_gpu_driver/vbios.h",root/"extra/nv_gpu_driver/pci_exp_table.h"],
  ["-DRPC_MESSAGE_STRUCTURES","-DRPC_STRUCTURES","-I{}/src/nvidia/generated","-I{}/src/common/inc","-I{}/src/nvidia/inc","-I{}/src/nvidia/interface/",
  "-I{}/src/nvidia/inc/kernel","-I{}/src/nvidia/inc/libraries","-I{}/src/nvidia/arch/nvalloc/common/inc","-I{}/kernel-open/nvidia-uvm",
  "-I{}/kernel-open/common/inc","-I{}/src/common/sdk/nvidia/inc","-I{}/src/nvidia/arch/nvalloc/unix/include","-I{}/src/common/sdk/nvidia/inc/ctrl"],
  tarball=nv_gpu.tarball)
    # this defines all syscall numbers. should probably unify linux autogen?
    case "io_uring": return load("io_uring",None,["/usr/include/liburing.h","/usr/include/linux/io_uring.h","/usr/include/asm-generic/unistd.h"],
                                 rules=[('__NR','NR')])
    case "ib": return load("ib", "ctypes.CDLL(ctypes.util.find_library('ibverbs'), use_errno=True)", ["/usr/include/infiniband/verbs.h",
  "/usr/include/infiniband/verbs_api.h","/usr/include/infiniband/ib_user_ioctl_verbs.h","/usr/include/rdma/ib_user_verbs.h"])
    case "llvm": return load("llvm","ctypes.CDLL(LLVM_PATH)",lambda:[sys("llvm-config-14 --includedir")+"/llvm-c/**/*.h"],
                             lambda: sys("llvm-config-14 --cflags"), recsym=True, prelude=["from tinygrad.runtime.support.llvm import LLVM_PATH"])
    case "pci": return load("pci", None, ["/usr/include/linux/pci_regs.h"])
    case "vfio": return load("vfio", None, ["/usr/include/linux/vfio.h"])
    # could add rule: WGPU_COMMA -> ','
    case "webgpu": return load("webgpu", "ctypes.CDLL(WEBGPU_PATH)",[root/"extra/webgpu/webgpu.h"],
                               prelude=["from tinygrad.runtime.support.webgpu import WEBGPU_PATH"])
    case "libusb": return load("libusb", "ctypes.CDLL(os.getenv('LIBUSB_PATH', ctypes.util.find_library('usb-1.0')))",
                               ["/usr/include/libusb-1.0/libusb.h"], prelude=["import os"])
    case _: raise AttributeError(f"no such autogen: {nm}")
