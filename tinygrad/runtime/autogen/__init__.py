import glob, importlib, pathlib, subprocess, tarfile
from tinygrad.helpers import fetch, flatten, system, getenv

root = (here:=pathlib.Path(__file__).parent).parents[2]
nv_src = {"nv_570": "https://github.com/NVIDIA/open-gpu-kernel-modules/archive/81fe4fb417c8ac3b9bdcc1d56827d116743892a5.tar.gz",
          "nv_580": "https://github.com/NVIDIA/open-gpu-kernel-modules/archive/2af9f1f0f7de4988432d4ae875b5858ffdb09cc2.tar.gz"}
macossdk = "/var/db/xcode_select_link/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"

def load(name, dll, files, **kwargs):
  if not (f:=(root/(path:=kwargs.pop("path", __name__)).replace('.','/')/f"{name}.py")).exists() or getenv('REGEN'):
    files, kwargs['args'] = files() if callable(files) else files, args() if callable(args:=kwargs.get('args', [])) else args
    if (tarball:=kwargs.pop('tarball', None)):
      # dangerous for arbitrary urls!
      with tarfile.open(fetch(tarball, gunzip=tarball.endswith("gz"))) as tf:
        tf.extractall("/tmp")
        base = f"/tmp/{tf.getnames()[0]}"
        files, kwargs['args'] = [str(f).format(base) for f in files], [a.format(base) for a in kwargs.get('args', [])]
        kwargs['anon_names'] = {k.format(base):v for k,v in kwargs.get('anon_names', {}).items()}
      if (preprocess:=kwargs.pop('preprocess', None)): preprocess(base)
    files = flatten(sorted(glob.glob(p, recursive=True)) if isinstance(p, str) and '*' in p else [p] for p in files)
    kwargs['epilog'] = (epi(base) if tarball else epi()) if callable(epi:=kwargs.get('epilog', [])) else epi
    f.write_text(importlib.import_module("tinygrad.runtime.support.autogen").gen(dll, files, **kwargs))
  return importlib.import_module(f"{path}.{name.replace('/', '.')}")

def __getattr__(nm):
  match nm:
    case "libc": return load("libc", ["find_library('c')"], lambda: (
      [i for i in system("dpkg -L libc6-dev").split() if 'sys/mman.h' in i or 'sys/syscall.h' in i] +
      ["/usr/include/string.h", "/usr/include/elf.h", "/usr/include/unistd.h", "/usr/include/asm-generic/mman-common.h"]), use_errno=True)
    case "opencl": return load("opencl", ["find_library('OpenCL')"], ["/usr/include/CL/cl.h"])
    case "cuda": return load("cuda", ["find_library('cuda')"], ["/usr/include/cuda.h"], args=["-D__CUDA_API_VERSION_INTERNAL"], parse_macros=False)
    case "nvrtc": return load("nvrtc", ["find_library('nvrtc')"], ["/usr/include/nvrtc.h"])
    case "nvjitlink": load("nvjitlink", ["find_library('nvJitLink')"], [root/"extra/nvJitLink.h"])
    case "kfd": return load("kfd", [], ["/usr/include/linux/kfd_ioctl.h"])
    case "nv_570" | "nv_580":
      return load(nm, [], [
        *[root/"extra/nv_gpu_driver"/s for s in ["clc6c0qmd.h","clcec0qmd.h"]], "{}/kernel-open/common/inc/nvmisc.h",
        *[f"{{}}/src/common/sdk/nvidia/inc/class/cl{s}.h" for s in ["0000", "0080", "2080", "2080_notification", "c56f", "c86f", "c96f", "c761",
                                                                    "83de", "c6c0", "cdc0"]],
        *[f"{{}}/kernel-open/nvidia-uvm/{s}.h" for s in ["clc6b5", "clc9b5", "uvm_ioctl", "uvm_linux_ioctl", "hwref/ampere/ga100/dev_fault"]],
        *[f"{{}}/src/nvidia/arch/nvalloc/unix/include/nv{s}.h" for s in ["_escape", "-ioctl", "-ioctl-numbers",
                                                                         "-ioctl-numa", "-unix-nvos-params-wrappers"]],
        *[f"{{}}/src/common/sdk/nvidia/inc/{s}.h" for s in ["alloc/alloc_channel", "nvos", "ctrl/ctrlc36f", "ctrl/ctrlcb33",
                                                            "ctrl/ctrla06c", "ctrl/ctrl90f1"]],
        *[f"{{}}/src/common/sdk/nvidia/inc/ctrl/ctrl{s}/*.h" for s in ["0000", "0080", "2080", "83de"]],
        "{}/kernel-open/common/inc/nvstatus.h", "{}/src/nvidia/generated/g_allclasses.h"
      ], args=[
        "-include", "{}/src/common/sdk/nvidia/inc/nvtypes.h", "-I{}/src/common/inc", "-I{}/kernel-open/nvidia-uvm", "-I{}/kernel-open/common/inc",
        "-I{}/src/common/sdk/nvidia/inc", "-I{}/src/nvidia/arch/nvalloc/unix/include", "-I{}/src/common/sdk/nvidia/inc/ctrl"
      ], rules=[(r'MW\(([^:]+):(.+)\)',r'(\1, \2)')], tarball=nv_src[nm], anon_names={"{}/kernel-open/common/inc/nvstatus.h:37":"nv_status_codes"})
    case "nv": return load("nv", [], [
      *[f"{{}}/src/nvidia/inc/kernel/gpu/{s}.h" for s in ["fsp/kern_fsp_cot_payload", "gsp/gsp_init_args"]],
      *[f"{{}}/src/nvidia/arch/nvalloc/common/inc/{s}.h" for s in ["gsp/gspifpub", "gsp/gsp_fw_wpr_meta", "gsp/gsp_fw_sr_meta", "rmRiscvUcode",
                                                                   "fsp/fsp_nvdm_format"]],
      *[f"{{}}/src/nvidia/inc/kernel/vgpu/{s}.h" for s in ["rpc_headers", "rpc_global_enums"]],
      "{}/src/common/uproc/os/common/include/libos_init_args.h", "{}/src/common/shared/msgq/inc/msgq/msgq_priv.h",
      "{}/src/nvidia/generated/g_rpc-structures.h", root/"extra/nv_gpu_driver/g_rpc-message-header.h", root/"extra/nv_gpu_driver/gsp_static_config.h",
      root/"extra/nv_gpu_driver/vbios.h", root/"extra/nv_gpu_driver/pci_exp_table.h"
    ], args=[
      "-DRPC_MESSAGE_STRUCTURES", "-DRPC_STRUCTURES", "-include", "{}/src/common/sdk/nvidia/inc/nvtypes.h", "-I{}/src/nvidia/generated",
      "-I{}/src/common/inc", "-I{}/src/nvidia/inc", "-I{}/src/nvidia/interface/", "-I{}/src/nvidia/inc/kernel", "-I{}/src/nvidia/inc/libraries",
      "-I{}/src/nvidia/arch/nvalloc/common/inc", "-I{}/kernel-open/nvidia-uvm", "-I{}/kernel-open/common/inc", "-I{}/src/common/sdk/nvidia/inc",
      "-I{}/src/nvidia/arch/nvalloc/unix/include", "-I{}/src/common/sdk/nvidia/inc/ctrl"
    ], tarball=nv_src["nv_570"], anon_names={
      "{}/src/nvidia/inc/kernel/vgpu/rpc_global_enums.h:8": "rpc_fns",
      "{}/src/nvidia/inc/kernel/vgpu/rpc_global_enums.h:244": "rpc_events"
    })
    # this defines all syscall numbers. should probably unify linux autogen?
    case "io_uring": return load("io_uring", [], ["/usr/include/liburing.h", "/usr/include/linux/io_uring.h", "/usr/include/asm-generic/unistd.h"],
                                 rules=[('__NR', 'NR')])
    case "ib": return load("ib", ["ibverbs"], ["/usr/include/infiniband/verbs.h", "/usr/include/infiniband/verbs_api.h",
                                               "/usr/include/infiniband/ib_user_ioctl_verbs.h","/usr/include/rdma/ib_user_verbs.h"], use_errno=True)
    case "llvm": return load("llvm", ["LLVM_PATH"], lambda: [system("llvm-config-20 --includedir")+"/llvm-c/**/*.h"],
                             args=lambda: system("llvm-config-20 --cflags").split(), recsym=True,
                             prolog=["from tinygrad.runtime.support.llvm import LLVM_PATH"])
    case "pci": return load("pci", [], ["/usr/include/linux/pci_regs.h"])
    case "vfio": return load("vfio", [], ["/usr/include/linux/vfio.h"])
    # could add rule: WGPU_COMMA -> ','
    case "webgpu":
      return load("webgpu", ["WEBGPU_PATH"], [root/"extra/webgpu/webgpu.h"], prolog=["from tinygrad.runtime.support.webgpu import WEBGPU_PATH"])
    case "libusb": return load("libusb", ["os.getenv('LIBUSB_PATH', find_library('usb-1.0'))"], ["/usr/include/libusb-1.0/libusb.h"])
    case "hip": return load("hip", ["os.getenv('ROCM_PATH', '/opt/rocm')+'/lib/libamdhip64.so'"], ["/opt/rocm/include/hip/hip_ext.h",
                            "/opt/rocm/include/hip/hiprtc.h", "/opt/rocm/include/hip/hip_runtime_api.h", "/opt/rocm/include/hip/driver_types.h"],
                            args=["-D__HIP_PLATFORM_AMD__", "-I/opt/rocm/include", "-x", "c++"])
    case "comgr" | "comgr_3":
      return load("comgr_3" if nm == "comgr_3" else "comgr", [
        "os.getenv('ROCM_PATH', '/opt/rocm')+'/lib/libamd_comgr.so'", "'/usr/local/lib/libamd_comgr.dylib'", "'/opt/homebrew/lib/libamd_comgr.dylib'"
      ], ["/opt/rocm/include/amd_comgr/amd_comgr.h"], args=["-D__HIP_PLATFORM_AMD__", "-I/opt/rocm/include", "-x", "c++"])
    case "hsa": return load("hsa", ["os.getenv('ROCM_PATH', '/opt/rocm')+'/lib/libhsa-runtime64.so'", "find_library('hsa-runtime64')"], [
      f"/opt/rocm/include/hsa/{s}.h" for s in ["hsa", "hsa_ext_amd", "amd_hsa_signal", "amd_hsa_queue", "amd_hsa_kernel_code", "hsa_ext_finalize",
                                               "hsa_ext_image", "hsa_ven_amd_aqlprofile"] ], args=["-I/opt/rocm/include"])
    case "amd_gpu": return load("amd_gpu", [], [root/f"extra/hip_gpu_driver/{s}.h" for s in ["sdma_registers", "nvd", "gc_11_0_0_offset",
                                                                                             "sienna_cichlid_ip_offset"]],
                                args=["-I/opt/rocm/include", "-x", "c++"])
    case "kgsl": return load("kgsl", [], [root/"extra/qcom_gpu_driver/msm_kgsl.h"], args=["-D__user="])
    case "adreno": return load("adreno", [], [root/"extra/qcom_gpu_driver/a6xx.xml.h"])
    case "qcom_dsp":
      return load("qcom_dsp", [], [root/f"extra/dsp/include/{s}.h" for s in ["ion", "msm_ion", "adsprpc_shared", "remote_default", "apps_std"]])
    case "sqtt": return load("sqtt", [], [root/"extra/sqtt/sqtt.h"])
    case "rocprof":
      return load("rocprof", ["find_library('rocprof-trace-decoder')", p:="'/usr/local/lib/rocprof-trace-decoder.so'", p.replace('so','dylib')],
                  [f"{{}}/include/{s}.h" for s in ["rocprof_trace_decoder", "trace_decoder_instrument", "trace_decoder_types"]],
                  tarball="https://github.com/ROCm/rocprof-trace-decoder/archive/dd0485100971522cc4cd8ae136bdda431061a04d.tar.gz")
    case "mesa": return load("mesa", ["find_library('tinymesa_cpu')",
      "(BASE:=os.getenv('MESA_PATH', f\"/usr{'/local/' if OSX else '/'}lib\"))+'/libtinymesa_cpu'+(EXT:='.dylib' if OSX else '.so')",
      "f'{BASE}/libtinymesa{EXT}'", "'/opt/homebrew/lib/libtinymesa_cpu.dylib'", "'/opt/homebrew/lib/libtinymesa.dylib'"], [
        *[f"{{}}/src/compiler/nir/{s}.h" for s in ["nir", "nir_builder", "nir_shader_compiler_options", "nir_serialize"]], "{}/gen/nir_intrinsics.h",
        *[f"{{}}/src/nouveau/{s}.h" for s in ["headers/nv_device_info", "compiler/nak"]],
        *[f"{{}}/src/gallium/auxiliary/gallivm/lp_bld{s}.h" for s in ["", "_passmgr", "_misc", "_type", "_init", "_nir", "_struct", "_jit_types",
                                                                     "_flow", "_const"]],
        "{}/src/compiler/glsl_types.h", "{}/src/util/blob.h", "{}/src/util/ralloc.h"], args=lambda:[
          "-DHAVE_ENDIAN_H", "-DHAVE_STRUCT_TIMESPEC", "-DHAVE_PTHREAD", "-DHAVE_FUNC_ATTRIBUTE_PACKED", "-I{}/src", "-I{}/include", "-I{}/gen",
          "-I{}/src/compiler/nir", "-I{}/src/gallium/auxiliary", "-I{}/src/gallium/include", f"-I{system('llvm-config-20 --includedir')}"],
        preprocess=lambda path: subprocess.run("""mkdir -p gen/util/format
python3 src/util/format/u_format_table.py src/util/format/u_format.yaml --enums > gen/util/format/u_format_gen.h
python3 src/compiler/nir/nir_opcodes_h.py > gen/nir_opcodes.h
python3 src/compiler/nir/nir_intrinsics_h.py --outdir gen
python3 src/compiler/nir/nir_intrinsics_indices_h.py --outdir gen
python3 src/compiler/nir/nir_builder_opcodes_h.py > gen/nir_builder_opcodes.h
python3 src/compiler/nir/nir_intrinsics_h.py --outdir gen
python3 src/compiler/builtin_types_h.py gen/builtin_types.h""", cwd=path, shell=True, check=True),
  tarball="https://gitlab.freedesktop.org/mesa/mesa/-/archive/mesa-25.2.4/mesa-25.2.4.tar.gz",
  prolog=["import gzip, base64", "from tinygrad.helpers import OSX"], epilog=lambda path: [system(f"{root}/extra/mesa/lvp_nir_options.sh {path}")])
    case "libclang":
      return load("libclang", ["os.getenv('LIBCLANG_PATH', find_library('clang-20'))"],
                  lambda: [f"{system('llvm-config-20 --includedir')}/clang-c/{s}.h" for s in ["Index", "CXString", "CXSourceLocation", "CXFile"]],
                  args=lambda: system("llvm-config-20 --cflags").split())
    case "metal":
      return load("metal", ["find_library('Metal')"],[f"{macossdk}/System/Library/Frameworks/Metal.framework/Headers/MTL{s}.h" for s in
                  ["ComputeCommandEncoder", "ComputePipeline", "CommandQueue", "Device", "IndirectCommandBuffer", "Resource", "CommandEncoder"]],
                  args=["-xobjective-c","-isysroot",macossdk], types={"dispatch_data_t":"objc.id_"})
    case _: raise AttributeError(f"no such autogen: {nm}")
