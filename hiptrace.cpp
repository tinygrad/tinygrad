// hiptrace.cpp
#define _GNU_SOURCE
#include <hip/hip_runtime_api.h>

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <atomic>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cxxabi.h>

static std::atomic<uint64_t> g_dump_seq{0};

static uint64_t get_tid() {
  return (uint64_t)syscall(SYS_gettid);
}

static void mkdir_p(const char* dir) {
  if (!dir || !dir[0]) return;
  mkdir(dir, 0755); // best-effort
}

static std::string sanitize(const char* s) {
  if (!s) return "unknown";
  std::string out;
  out.reserve(128);
  for (const char* p = s; *p; p++) {
    char c = *p;
    bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
    out.push_back(ok ? c : '_');
    if (out.size() >= 120) break;
  }
  if (out.empty()) out = "unknown";
  return out;
}

static void dump_bytes_to_file(const char* tag, const char* kname, const void* buf, size_t n) {
  const char* dir = "trace";

  mkdir_p(dir);

  uint64_t seq = g_dump_seq.fetch_add(1, std::memory_order_relaxed);
  pid_t pid = getpid();
  uint64_t tid = get_tid();

  std::string sk = sanitize(kname);
  char path[4096];
  snprintf(path, sizeof(path), "%s/%d.%llu.%llu.%s.%s.kernargs.bin",
           dir, (int)pid,
           (unsigned long long)tid,
           (unsigned long long)seq,
           tag ? tag : "launch",
           sk.c_str());

  FILE* f = fopen(path, "wb");
  if (!f) return;
  fwrite(buf, 1, n, f);
  fclose(f);

  // optional: print where it went (toggle with HIPTRACE_DUMP_PRINT=1)
  fprintf(stderr, "[hiptrace] wrote %zu bytes -> %s\n", n, path);
}

static int env_int(const char* k, int dflt) {
  const char* v = getenv(k);
  return v ? atoi(v) : dflt;
}

static std::string demangle(const char* name) {
  if (!name) return "<unknown>";
  int status = 0;
  size_t n = 0;
  char* out = abi::__cxa_demangle(name, nullptr, &n, &status);
  std::string s = (status == 0 && out) ? out : name;
  free(out);
  return s;
}

static std::string sym_name_from_ptr(const void* p) {
  if (!p) return "<null>";
  Dl_info info;
  if (dladdr(p, &info) && info.dli_sname) return demangle(info.dli_sname);
  return "<no-symbol>";
}

static void hexdump(const void* p, size_t n, size_t max_bytes) {
  if (!p) {
    fprintf(stderr, "  [argbuf] <null>\n");
    return;
  }
  size_t m = n < max_bytes ? n : max_bytes;
  const unsigned char* b = (const unsigned char*)p;
  fprintf(stderr, "  [argbuf] %zu bytes (showing %zu):\n", n, m);
  for (size_t i = 0; i < m; i += 16) {
    fprintf(stderr, "    %04zx: ", i);
    for (size_t j = 0; j < 16; j++) {
      if (i + j < m) fprintf(stderr, "%02x ", b[i + j]);
      else fprintf(stderr, "   ");
    }
    fprintf(stderr, " |");
    for (size_t j = 0; j < 16 && i + j < m; j++) {
      unsigned char c = b[i + j];
      fprintf(stderr, "%c", (c >= 32 && c <= 126) ? c : '.');
    }
    fprintf(stderr, "|\n");
  }
  if (m < n) fprintf(stderr, "  [argbuf] ... truncated (set HIPTRACE_MAX_DUMP)\n");
}

// Known kernel arg sizes from ELF metadata
static size_t get_kernarg_size(const char* kname) {
  if (!kname) return 0;
  // Forward kernel
  if (strstr(kname, "fmha_fwd_hd128_bf16_causal")) return 512;
  // Backward kernels
  if (strstr(kname, "fmha_bwd_hd128_odo_bf16")) return 208;
  if (strstr(kname, "fmha_bwd_hd128_bf16_causal_br_a32_psskddv")) return 704;
  if (strstr(kname, "fmha_bwd_hd128_dq_convert_bf16")) return 208;
  return 0;
}

static void dump_params_or_extra(void** kernelParams, void** extra, const char* kname) {
  const void* argbuf = nullptr;

  if (extra) {
    for (int i = 0; extra[i] != HIP_LAUNCH_PARAM_END; i += 2) {
      if (extra[i] == HIP_LAUNCH_PARAM_BUFFER_POINTER) {
        argbuf = extra[i + 1];
      }
    }
  }

  size_t argbuf_size = get_kernarg_size(kname);
  fprintf(stderr, "  [extra] argbuf=%p size=%zu (from ELF)\n", argbuf, argbuf_size);

  if (argbuf && argbuf_size) {
    dump_bytes_to_file("kernargs", kname ? kname : "unknown", argbuf, argbuf_size);
  }

  if (kernelParams) {
    int show = env_int("HIPTRACE_SHOW_ARG_SLOTS", 8);
    fprintf(stderr, "  [kernelParams] host arg slots (unknown types), showing %d:\n", show);
    for (int i = 0; i < show; i++) fprintf(stderr, "    kernelParams[%d]=%p\n", i, (void*)kernelParams[i]);
  }
}

struct ReentryGuard {
  static thread_local int depth;
  bool active = false;
  ReentryGuard() {
    if (depth == 0) active = true;
    depth++;
  }
  ~ReentryGuard() { depth--; }
  operator bool() const { return active; }
};
thread_local int ReentryGuard::depth = 0;

// Per-thread captured args for hipConfigureCall/hipSetupArgument/hipLaunchByPtr path
struct PendingArgs {
  dim3 grid{0, 0, 0};
  dim3 block{0, 0, 0};
  size_t shmem = 0;
  hipStream_t stream = nullptr;
  std::vector<unsigned char> buf;
  bool active = false;
};
static thread_local PendingArgs g_pending;

// Map hipFunction_t -> kernel name (from hipModuleGetFunction)
static std::mutex g_mu;
static std::unordered_map<void*, std::string> g_func_names;

static const char* func_name_for(hipFunction_t f) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto it = g_func_names.find((void*)f);
  return (it == g_func_names.end()) ? nullptr : it->second.c_str();
}

// Real symbols
static hipError_t (*real_hipLaunchKernel)(const void*, dim3, dim3, void**, size_t, hipStream_t) = nullptr;
static hipError_t (*real_hipLaunchKernel_spt)(const void*, dim3, dim3, void**, size_t, hipStream_t) = nullptr;

static hipError_t (*real_hipModuleGetFunction)(hipFunction_t*, hipModule_t, const char*) = nullptr;
static hipError_t (*real_hipModuleLaunchKernel)(
  hipFunction_t,
  unsigned int, unsigned int, unsigned int,
  unsigned int, unsigned int, unsigned int,
  unsigned int, hipStream_t, void**, void**) = nullptr;

static hipError_t (*real_hipExtModuleLaunchKernel)(
  hipFunction_t,
  uint32_t, uint32_t, uint32_t,
  uint32_t, uint32_t, uint32_t,
  size_t, hipStream_t, void**, void**,
  hipEvent_t, hipEvent_t, uint32_t) = nullptr;

static hipError_t (*real_hipExtLaunchKernel)(
  const void*, dim3, dim3, void**, size_t, hipStream_t,
  hipEvent_t, hipEvent_t, int) = nullptr;

static hipError_t (*real_hipConfigureCall)(dim3, dim3, size_t, hipStream_t) = nullptr;
static hipError_t (*real_hipSetupArgument)(const void*, size_t, size_t) = nullptr;
static hipError_t (*real_hipLaunchByPtr)(const void*) = nullptr;

static void resolve_syms() {
  if (!real_hipLaunchKernel)
    real_hipLaunchKernel =
      (hipError_t (*)(const void*, dim3, dim3, void**, size_t, hipStream_t))dlsym(RTLD_NEXT, "hipLaunchKernel");

  if (!real_hipLaunchKernel_spt)
    real_hipLaunchKernel_spt =
      (hipError_t (*)(const void*, dim3, dim3, void**, size_t, hipStream_t))dlsym(RTLD_NEXT, "hipLaunchKernel_spt");

  if (!real_hipModuleGetFunction)
    real_hipModuleGetFunction =
      (hipError_t (*)(hipFunction_t*, hipModule_t, const char*))dlsym(RTLD_NEXT, "hipModuleGetFunction");

  if (!real_hipModuleLaunchKernel)
    real_hipModuleLaunchKernel =
      (hipError_t (*)(
        hipFunction_t,
        unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int,
        unsigned int, hipStream_t, void**, void**))dlsym(RTLD_NEXT, "hipModuleLaunchKernel");

  if (!real_hipExtModuleLaunchKernel)
    real_hipExtModuleLaunchKernel =
      (hipError_t (*)(
        hipFunction_t,
        uint32_t, uint32_t, uint32_t,
        uint32_t, uint32_t, uint32_t,
        size_t, hipStream_t, void**, void**,
        hipEvent_t, hipEvent_t, uint32_t))dlsym(RTLD_NEXT, "hipExtModuleLaunchKernel");

  if (!real_hipExtLaunchKernel)
    real_hipExtLaunchKernel =
      (hipError_t (*)(
        const void*, dim3, dim3, void**, size_t, hipStream_t,
        hipEvent_t, hipEvent_t, int))dlsym(RTLD_NEXT, "hipExtLaunchKernel");

  if (!real_hipConfigureCall)
    real_hipConfigureCall =
      (hipError_t (*)(dim3, dim3, size_t, hipStream_t))dlsym(RTLD_NEXT, "hipConfigureCall");

  if (!real_hipSetupArgument)
    real_hipSetupArgument =
      (hipError_t (*)(const void*, size_t, size_t))dlsym(RTLD_NEXT, "hipSetupArgument");

  if (!real_hipLaunchByPtr)
    real_hipLaunchByPtr =
      (hipError_t (*)(const void*))dlsym(RTLD_NEXT, "hipLaunchByPtr");
}

// ---- Interposers ----

extern "C" hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname) {
  resolve_syms();
  if (!real_hipModuleGetFunction) {
    fprintf(stderr, "[hiptrace] ERROR: could not resolve real hipModuleGetFunction\n");
    return hipErrorUnknown;
  }

  hipError_t st = real_hipModuleGetFunction(function, module, kname);

  if (st == hipSuccess && function && *function && kname) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_func_names[(void*)(*function)] = kname;
  }

  fprintf(stderr, "[hiptrace] hipModuleGetFunction(module=%p, name=%s) -> f=%p status=%d\n",
          (void*)module, kname ? kname : "<null>", function ? (void*)(*function) : nullptr, (int)st);
  return st;
}

extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t f,
                                           unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                           unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                           unsigned int sharedMemBytes, hipStream_t stream,
                                           void** kernelParams, void** extra) {
  resolve_syms();
  if (!real_hipModuleLaunchKernel) {
    fprintf(stderr, "[hiptrace] ERROR: could not resolve real hipModuleLaunchKernel\n");
    return hipErrorUnknown;
  }

  ReentryGuard g;
  if (!g) return real_hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                           blockDimX, blockDimY, blockDimZ,
                                           sharedMemBytes, stream, kernelParams, extra);

  const char* known = func_name_for(f);
  fprintf(stderr,
          "[hiptrace] hipModuleLaunchKernel(f=%p name=%s grid=(%u,%u,%u) block=(%u,%u,%u) shmem=%u stream=%p)\n",
          (void*)f, known ? known : "<unknown>",
          gridDimX, gridDimY, gridDimZ,
          blockDimX, blockDimY, blockDimZ,
          sharedMemBytes, (void*)stream);

  dump_params_or_extra(kernelParams, extra, known);

  hipError_t st = real_hipModuleLaunchKernel(f,
                                            gridDimX, gridDimY, gridDimZ,
                                            blockDimX, blockDimY, blockDimZ,
                                            sharedMemBytes, stream,
                                            kernelParams, extra);
  fprintf(stderr, "[hiptrace] hipModuleLaunchKernel -> status=%d\n", (int)st);
  return st;
}

static bool extract_extra_argbuf(void** extra, const void** argbuf_out, size_t* sz_out) {
  const void* argbuf = nullptr;
  size_t sz = 0;
  if (extra) {
    for (int i = 0; extra[i] != HIP_LAUNCH_PARAM_END; i += 2) {
      if (extra[i] == HIP_LAUNCH_PARAM_BUFFER_POINTER) argbuf = extra[i + 1];
      else if (extra[i] == HIP_LAUNCH_PARAM_BUFFER_SIZE && extra[i + 1]) sz = *(size_t*)extra[i + 1];
    }
  }
  if (argbuf_out) *argbuf_out = argbuf;
  if (sz_out) *sz_out = sz;
  return argbuf && sz;
}

static const char* which_args(void** kernelParams, void** extra) {
  const void* argbuf = nullptr;
  size_t sz = 0;
  if (extract_extra_argbuf(extra, &argbuf, &sz)) return "extra(packed)";
  if (kernelParams) return "kernelParams(ptrs)";
  if (extra) return "extra(unknown)";
  return "none";
}

extern "C" hipError_t hipExtModuleLaunchKernel(
  hipFunction_t f,
  uint32_t globalWorkSizeX, uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
  uint32_t localWorkSizeX,  uint32_t localWorkSizeY,  uint32_t localWorkSizeZ,
  size_t sharedMemBytes,
  hipStream_t hStream,
  void** kernelParams,
  void** extra,
  hipEvent_t startEvent,
  hipEvent_t stopEvent,
  uint32_t flags) {
  resolve_syms();
  if (!real_hipExtModuleLaunchKernel) {
    fprintf(stderr, "[hiptrace] ERROR: could not resolve real hipExtModuleLaunchKernel\n");
    return hipErrorUnknown;
  }

  ReentryGuard g;
  if (!g) {
    return real_hipExtModuleLaunchKernel(
      f,
      globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
      localWorkSizeX, localWorkSizeY, localWorkSizeZ,
      sharedMemBytes, hStream, kernelParams, extra,
      startEvent, stopEvent, flags);
  }

  const char* known = func_name_for(f);
  fprintf(stderr, "  [hiptrace] argmode=%s\n", which_args(kernelParams, extra));
  fprintf(stderr,
          "[hiptrace] hipExtModuleLaunchKernel(f=%p name=%s global=(%u,%u,%u) local=(%u,%u,%u) shmem=%zu stream=%p start=%p stop=%p flags=0x%x)\n",
          (void*)f, known ? known : "<unknown>",
          globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
          localWorkSizeX, localWorkSizeY, localWorkSizeZ,
          sharedMemBytes, (void*)hStream, (void*)startEvent, (void*)stopEvent, flags);

  dump_params_or_extra(kernelParams, extra, known);

  hipError_t st = real_hipExtModuleLaunchKernel(
    f,
    globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
    localWorkSizeX, localWorkSizeY, localWorkSizeZ,
    sharedMemBytes, hStream, kernelParams, extra,
    startEvent, stopEvent, flags);

  fprintf(stderr, "[hiptrace] hipExtModuleLaunchKernel -> status=%d\n", (int)st);
  return st;
}

extern "C" hipError_t hipLaunchKernel(const void* func,
                                     dim3 gridDim,
                                     dim3 blockDim,
                                     void** args,
                                     size_t sharedMemBytes,
                                     hipStream_t stream) {
  resolve_syms();
  if (!real_hipLaunchKernel) {
    fprintf(stderr, "[hiptrace] ERROR: could not resolve real hipLaunchKernel\n");
    return hipErrorUnknown;
  }

  ReentryGuard g;
  if (!g) return real_hipLaunchKernel(func, gridDim, blockDim, args, sharedMemBytes, stream);

  std::string kname = sym_name_from_ptr(func);
  fprintf(stderr,
          "[hiptrace] hipLaunchKernel(func=%p %s, grid=(%u,%u,%u) block=(%u,%u,%u) shmem=%zu stream=%p)\n",
          func, kname.c_str(),
          gridDim.x, gridDim.y, gridDim.z,
          blockDim.x, blockDim.y, blockDim.z,
          sharedMemBytes, (void*)stream);

  if (args) {
    int show = env_int("HIPTRACE_SHOW_ARG_SLOTS", 8);
    fprintf(stderr, "  [args] host arg slots (unknown types), showing %d:\n", show);
    for (int i = 0; i < show; i++) fprintf(stderr, "    args[%d]=%p\n", i, (void*)args[i]);
  }

  hipError_t st = real_hipLaunchKernel(func, gridDim, blockDim, args, sharedMemBytes, stream);
  fprintf(stderr, "[hiptrace] hipLaunchKernel -> status=%d\n", (int)st);
  return st;
}

extern "C" hipError_t hipLaunchKernel_spt(const void* func,
                                         dim3 gridDim,
                                         dim3 blockDim,
                                         void** args,
                                         size_t sharedMemBytes,
                                         hipStream_t stream) {
  resolve_syms();
  if (!real_hipLaunchKernel_spt) {
    // If the runtime doesn't export it on your ROCm, just fall back if possible.
    if (real_hipLaunchKernel) return hipLaunchKernel(func, gridDim, blockDim, args, sharedMemBytes, stream);
    fprintf(stderr, "[hiptrace] ERROR: could not resolve real hipLaunchKernel_spt\n");
    return hipErrorUnknown;
  }

  ReentryGuard g;
  if (!g) return real_hipLaunchKernel_spt(func, gridDim, blockDim, args, sharedMemBytes, stream);

  std::string kname = sym_name_from_ptr(func);
  fprintf(stderr,
          "[hiptrace] hipLaunchKernel_spt(func=%p %s, grid=(%u,%u,%u) block=(%u,%u,%u) shmem=%zu stream=%p)\n",
          func, kname.c_str(),
          gridDim.x, gridDim.y, gridDim.z,
          blockDim.x, blockDim.y, blockDim.z,
          sharedMemBytes, (void*)stream);

  if (args) {
    int show = env_int("HIPTRACE_SHOW_ARG_SLOTS", 8);
    fprintf(stderr, "  [args] host arg slots (unknown types), showing %d:\n", show);
    for (int i = 0; i < show; i++) fprintf(stderr, "    args[%d]=%p\n", i, (void*)args[i]);
  }

  hipError_t st = real_hipLaunchKernel_spt(func, gridDim, blockDim, args, sharedMemBytes, stream);
  fprintf(stderr, "[hiptrace] hipLaunchKernel_spt -> status=%d\n", (int)st);
  return st;
}

extern "C" hipError_t hipExtLaunchKernel(const void* function_address,
                                        dim3 numBlocks,
                                        dim3 dimBlocks,
                                        void** args,
                                        size_t sharedMemBytes,
                                        hipStream_t stream,
                                        hipEvent_t startEvent,
                                        hipEvent_t stopEvent,
                                        int flags) {
  resolve_syms();
  if (!real_hipExtLaunchKernel) {
    fprintf(stderr, "[hiptrace] ERROR: could not resolve real hipExtLaunchKernel\n");
    return hipErrorUnknown;
  }

  ReentryGuard g;
  if (!g) return real_hipExtLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream, startEvent, stopEvent, flags);

  std::string kname = sym_name_from_ptr(function_address);
  fprintf(stderr,
          "[hiptrace] hipExtLaunchKernel(func=%p %s, grid=(%u,%u,%u) block=(%u,%u,%u) shmem=%zu stream=%p start=%p stop=%p flags=0x%x)\n",
          function_address, kname.c_str(),
          numBlocks.x, numBlocks.y, numBlocks.z,
          dimBlocks.x, dimBlocks.y, dimBlocks.z,
          sharedMemBytes, (void*)stream, (void*)startEvent, (void*)stopEvent, flags);

  if (args) {
    int show = env_int("HIPTRACE_SHOW_ARG_SLOTS", 8);
    fprintf(stderr, "  [args] host arg slots (unknown types), showing %d:\n", show);
    for (int i = 0; i < show; i++) fprintf(stderr, "    args[%d]=%p\n", i, (void*)args[i]);
  }

  hipError_t st = real_hipExtLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream, startEvent, stopEvent, flags);
  fprintf(stderr, "[hiptrace] hipExtLaunchKernel -> status=%d\n", (int)st);
  return st;
}

// ---- Stub-style argument packing path ----

extern "C" hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream) {
  resolve_syms();
  if (!real_hipConfigureCall) return hipErrorUnknown;

  g_pending.grid = gridDim;
  g_pending.block = blockDim;
  g_pending.shmem = sharedMem;
  g_pending.stream = stream;
  g_pending.buf.clear();
  g_pending.active = true;

  return real_hipConfigureCall(gridDim, blockDim, sharedMem, stream);
}

extern "C" hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset) {
  resolve_syms();
  if (!real_hipSetupArgument) return hipErrorUnknown;

  if (g_pending.active) {
    size_t need = offset + size;
    if (g_pending.buf.size() < need) g_pending.buf.resize(need);
    if (arg && size) memcpy(g_pending.buf.data() + offset, arg, size);

    if (env_int("HIPTRACE_LOG_SETUPARG", 0)) {
      fprintf(stderr, "[hiptrace] hipSetupArgument(arg=%p size=%zu offset=%zu)\n", arg, size, offset);
    }
  }

  return real_hipSetupArgument(arg, size, offset);
}

extern "C" hipError_t hipLaunchByPtr(const void* func) {
  resolve_syms();
  if (!real_hipLaunchByPtr) return hipErrorUnknown;

  ReentryGuard g;
  if (!g) return real_hipLaunchByPtr(func);

  std::string kname = sym_name_from_ptr(func);
  fprintf(stderr,
          "[hiptrace] hipLaunchByPtr(func=%p %s, grid=(%u,%u,%u) block=(%u,%u,%u) shmem=%zu stream=%p)\n",
          func, kname.c_str(),
          g_pending.grid.x, g_pending.grid.y, g_pending.grid.z,
          g_pending.block.x, g_pending.block.y, g_pending.block.z,
          g_pending.shmem, (void*)g_pending.stream);

  if (g_pending.active && !g_pending.buf.empty()) {
    size_t max_dump = (size_t)env_int("HIPTRACE_MAX_DUMP", 256);
    dump_bytes_to_file("by_ptr", kname.c_str(), g_pending.buf.data(), g_pending.buf.size());
  }

  hipError_t st = real_hipLaunchByPtr(func);
  fprintf(stderr, "[hiptrace] hipLaunchByPtr -> status=%d\n", (int)st);

  g_pending.active = false;
  return st;
}
