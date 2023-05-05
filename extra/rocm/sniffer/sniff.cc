// template copied from https://github.com/geohot/cuda_ioctl_sniffer/blob/master/sniff.cc

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <signal.h>
#include <ucontext.h>

#include <sys/mman.h>

// includes from the ROCm sources
#include <linux/kfd_ioctl.h>
#include <hsa.h>
#include <amd_hsa_kernel_code.h>

#include <string>
#include <map>
std::map<int, std::string> files;
std::map<uint64_t, uint64_t> ring_base_addresses;

#define D(args...) fprintf(stderr, args)

uint64_t doorbell_offset = -1;
int queue_type = 0;

void hexdump(void *d, int l) {
  for (int i = 0; i < l; i++) {
    if (i%0x10 == 0 && i != 0) printf("\n");
    if (i%0x10 == 8) printf(" ");
    if (i%0x10 == 0) printf("%8X: ", i);
    printf("%2.2X ", ((uint8_t*)d)[i]);
  }
  printf("\n");
}

extern "C" {

// https://defuse.ca/online-x86-assembler.htm#disassembly2
static void handler(int sig, siginfo_t *si, void *unused) {
  ucontext_t *u = (ucontext_t *)unused;
  uint8_t *rip = (uint8_t*)u->uc_mcontext.gregs[REG_RIP];

  int store_size = 0;
  uint64_t value;
  if (rip[0] == 0x48 && rip[1] == 0x89 && rip[2] == 0x30) {
    // 0:  48 89 30                mov    QWORD PTR [rax],rsi
    store_size = 8;
    value = u->uc_mcontext.gregs[REG_RSI];
    u->uc_mcontext.gregs[REG_RIP] += 3;
  } else if (rip[0] == 0x4c && rip[1] == 0x89 && rip[2] == 0x28) {
    // 0:  4c 89 28                mov    QWORD PTR [rax],r13
    store_size = 8;
    value = u->uc_mcontext.gregs[REG_R13];
    u->uc_mcontext.gregs[REG_RIP] += 3;
  } else {
    D("segfault %02X %02X %02X %02X %02X %02X %02X %02X rip: %p addr: %p\n", rip[0], rip[1], rip[2], rip[3], rip[4], rip[5], rip[6], rip[7], rip, si->si_addr);
    D("rax: %llx rcx: %llx rdx: %llx rsi: %llx rbx: %llx\n", u->uc_mcontext.gregs[REG_RAX], u->uc_mcontext.gregs[REG_RCX], u->uc_mcontext.gregs[REG_RDX], u->uc_mcontext.gregs[REG_RSI], u->uc_mcontext.gregs[REG_RBX]);
    exit(-1);
  }

  uint64_t ring_base_address = ring_base_addresses[((uint64_t)si->si_addr)&0xFFF];
  D("%16p: DING DONG store(%d): 0x%8lx -> %p ring_base_address:0x%lx\n", rip, store_size, value, si->si_addr, ring_base_address);

  if (queue_type == KFD_IOC_QUEUE_TYPE_SDMA) {
    hexdump((void*)(ring_base_address), 0x100);
  } else if (queue_type == KFD_IOC_QUEUE_TYPE_COMPUTE_AQL) {
    hexdump((void*)(ring_base_address+value*0x40), 0x40);

    hsa_kernel_dispatch_packet_t *pkt = (hsa_kernel_dispatch_packet_t *)(ring_base_address+value*0x40);
    if ((pkt->header&0xFF) == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
      D("HSA_PACKET_TYPE_KERNEL_DISPATCH -- setup:%d workgroup[%d, %d, %d] grid[%d, %d, %d] kernel_object:0x%lx kernarg_address:%p\n", pkt->setup, pkt->workgroup_size_x, pkt->workgroup_size_y, pkt->workgroup_size_z, pkt->grid_size_x, pkt->grid_size_y, pkt->grid_size_z, pkt->kernel_object, pkt->kernarg_address);
      amd_kernel_code_t *code = (amd_kernel_code_t *)pkt->kernel_object;
      D("kernel_code_entry_byte_offset:%lx\n", code->kernel_code_entry_byte_offset);
      hexdump((void*)(pkt->kernel_object + code->kernel_code_entry_byte_offset), 0x200);
      //hexdump((void*)pkt->kernel_object, sizeof(amd_kernel_code_t));
    } else if ((pkt->header&0xFF) == HSA_PACKET_TYPE_BARRIER_AND) {
      D("HSA_PACKET_TYPE_BARRIER_AND\n");
    }
  }

  mprotect((void *)((uint64_t)si->si_addr & ~0xFFF), 0x2000, PROT_READ | PROT_WRITE);
  if (store_size == 8) {
    *(volatile uint64_t*)(si->si_addr) = value;
  } else if (store_size == 4) {
    *(volatile uint32_t*)(si->si_addr) = value;
  } else if (store_size == 2) {
    *(volatile uint16_t*)(si->si_addr) = value;
  } else {
    D("store size not supported\n");
    exit(-1);
  }
  mprotect((void *)((uint64_t)si->si_addr & ~0xFFF), 0x2000, PROT_NONE);
}

void register_sigsegv_handler() {
  struct sigaction sa;
  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  sa.sa_sigaction = handler;
  sigaction(SIGSEGV, &sa, NULL);
}

int (*my_open)(const char *pathname, int flags, mode_t mode);
#undef open
int open(const char *pathname, int flags, mode_t mode) {
  if (my_open == NULL) my_open = reinterpret_cast<decltype(my_open)>(dlsym(RTLD_NEXT, "open"));
  int ret = my_open(pathname, flags, mode);
  //D("open %s (0o%o) = %d\n", pathname, flags, ret);
  files[ret] = pathname;
  return ret;
}


int (*my_open64)(const char *pathname, int flags, mode_t mode);
#undef open
int open64(const char *pathname, int flags, mode_t mode) {
  if (my_open64 == NULL) my_open64 = reinterpret_cast<decltype(my_open64)>(dlsym(RTLD_NEXT, "open64"));
  int ret = my_open64(pathname, flags, mode);
  //D("open %s (0o%o) = %d\n", pathname, flags, ret);
  files[ret] = pathname;
  return ret;
}

void *(*my_mmap)(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
#undef mmap
void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  if (my_mmap == NULL) my_mmap = reinterpret_cast<decltype(my_mmap)>(dlsym(RTLD_NEXT, "mmap"));
  void *ret = my_mmap(addr, length, prot, flags, fd, offset);

  if (doorbell_offset != -1 && offset == doorbell_offset) {
    D("HIDDEN DOORBELL %p\n", addr);
    register_sigsegv_handler();
    mprotect(addr, length, PROT_NONE);
  }

  if (fd != -1) D("mmapped %p (target %p) with flags 0x%x length 0x%zx fd %d %s offset 0x%lx\n", ret, addr, flags, length, fd, files[fd].c_str(), offset);
  return ret;
}

void *(*my_mmap64)(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
#undef mmap64
void *mmap64(void *addr, size_t length, int prot, int flags, int fd, off_t offset) { return mmap(addr, length, prot, flags, fd, offset); }

int ioctl_num = 1;
int (*my_ioctl)(int filedes, unsigned long request, void *argp) = NULL;
#undef ioctl
int ioctl(int filedes, unsigned long request, void *argp) {
  if (my_ioctl == NULL) my_ioctl = reinterpret_cast<decltype(my_ioctl)>(dlsym(RTLD_NEXT, "ioctl"));
  int ret = 0;
  ret = my_ioctl(filedes, request, argp);
  if (!files.count(filedes)) return ret;

  uint8_t type = (request >> 8) & 0xFF;
  uint8_t nr = (request >> 0) & 0xFF;
  uint16_t size = (request >> 16) & 0xFFF;

  D("%3d: %d = %3d(%20s) 0x%3x ", ioctl_num, ret, filedes, files[filedes].c_str(), size);

  if (request == AMDKFD_IOC_SET_EVENT) {
    kfd_ioctl_set_event_args *args = (kfd_ioctl_set_event_args *)argp;
    D("AMDKFD_IOC_SET_EVENT event_id:%d", args->event_id);
  } else if (request == AMDKFD_IOC_ALLOC_MEMORY_OF_GPU) {
    kfd_ioctl_alloc_memory_of_gpu_args *args = (kfd_ioctl_alloc_memory_of_gpu_args *)argp;
    D("AMDKFD_IOC_ALLOC_MEMORY_OF_GPU va_addr:0x%llx size:0x%llx handle:%llX gpu_id:0x%x", args->va_addr, args->size, args->handle, args->gpu_id);
  } else if (request == AMDKFD_IOC_MAP_MEMORY_TO_GPU) {
    kfd_ioctl_map_memory_to_gpu_args *args = (kfd_ioctl_map_memory_to_gpu_args *)argp;
    D("AMDKFD_IOC_MAP_MEMORY_TO_GPU handle:%llX", args->handle);
  } else if (request == AMDKFD_IOC_CREATE_EVENT) {
    kfd_ioctl_create_event_args *args = (kfd_ioctl_create_event_args *)argp;
    D("AMDKFD_IOC_CREATE_EVENT event_type:%d event_id:%d", args->event_type, args->event_id);
  } else if (request == AMDKFD_IOC_WAIT_EVENTS) {
    D("AMDKFD_IOC_WAIT_EVENTS");
  } else if (request == AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU) {
    kfd_ioctl_unmap_memory_from_gpu_args *args = (kfd_ioctl_unmap_memory_from_gpu_args *)argp;
    D("AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU handle:%llX", args->handle);
  } else if (request == AMDKFD_IOC_FREE_MEMORY_OF_GPU) {
    D("AMDKFD_IOC_FREE_MEMORY_OF_GPU");
  } else if (request == AMDKFD_IOC_SET_SCRATCH_BACKING_VA) {
    D("AMDKFD_IOC_SET_SCRATCH_BACKING_VA");
  } else if (request == AMDKFD_IOC_GET_TILE_CONFIG) {
    D("AMDKFD_IOC_GET_TILE_CONFIG");
  } else if (request == AMDKFD_IOC_SET_TRAP_HANDLER) {
    D("AMDKFD_IOC_SET_TRAP_HANDLER");
  } else if (request == AMDKFD_IOC_GET_VERSION) {
    kfd_ioctl_get_version_args *args = (kfd_ioctl_get_version_args *)argp;
    D("AMDKFD_IOC_GET_VERSION major_version:%d minor_version:%d", args->major_version, args->minor_version);
  } else if (request == AMDKFD_IOC_GET_PROCESS_APERTURES_NEW) {
    D("AMDKFD_IOC_GET_PROCESS_APERTURES_NEW");
  } else if (request == AMDKFD_IOC_ACQUIRE_VM) {
    D("AMDKFD_IOC_ACQUIRE_VM");
  } else if (request == AMDKFD_IOC_SET_MEMORY_POLICY) {
    D("AMDKFD_IOC_SET_MEMORY_POLICY");
  } else if (request == AMDKFD_IOC_GET_CLOCK_COUNTERS) {
    D("AMDKFD_IOC_GET_CLOCK_COUNTERS");
  } else if (request == AMDKFD_IOC_CREATE_QUEUE) {
    kfd_ioctl_create_queue_args *args = (kfd_ioctl_create_queue_args *)argp;
    D("AMDKFD_IOC_CREATE_QUEUE\n");
    D("queue_type:%d ring_base_address:0x%llx\n", args->queue_type, args->ring_base_address);
    D("eop_buffer_address:0x%llx ctx_save_restore_address:0x%llx\n", args->eop_buffer_address, args->ctx_save_restore_address);
    D("ring_size:0x%x queue_priority:%d\n", args->ring_size, args->queue_priority);
    D("RETURNS write_pointer_address:0x%llx read_pointer_address:0x%llx doorbell_offset:0x%llx queue_id:%d\n", args->write_pointer_address, args->read_pointer_address, args->doorbell_offset, args->queue_id);
    //D("RETURNS *write_pointer_address:0x%llx *read_pointer_address:0x%llx\n", *(uint64_t*)args->write_pointer_address, *(uint64_t*)args->read_pointer_address);
    ring_base_addresses[args->doorbell_offset&0xFFF] = args->ring_base_address;
    doorbell_offset = args->doorbell_offset&~0xFFF;
    queue_type = args->queue_type;
  } else {
    D("type:0x%x nr:0x%x size:0x%x", type, nr, size);
  }

  D("\n");
  ioctl_num++;
  return ret;
}

}
