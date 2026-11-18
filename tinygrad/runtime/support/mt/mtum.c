/* Moore Threads userspace launch-generation support. This is not a hardware execution shim. */
#define _GNU_SOURCE
#include <stdarg.h>
#include <stdio.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <dirent.h>
#include <string.h>
#include <stdint.h>
#include "version_data.h"

static int log_open(const char *path, int flags, mode_t mode, int kind) {
  static int (*real_open)(const char *, int, ...);
  static int (*real_openat)(int, const char *, int, ...);
  if (!real_open) real_open = dlsym(RTLD_NEXT, kind == 1 ? "open64" : "open");
  if (!real_openat) real_openat = dlsym(RTLD_NEXT, "openat");
  const char *actual = path[0] ? path : "/dev/dri/card0";
  int fd = kind == 2 ? real_openat(AT_FDCWD, actual, flags, mode) : real_open(actual, flags, mode);
  fprintf(stderr, "OPEN%d %s [%s] flags %#x = %d caller %p\n", kind, path, actual, flags, fd, __builtin_return_address(1));
  return fd;
}
int __xstat(int ver, const char *path, struct stat *st) { static int (*fn)(int,const char*,struct stat*); if (!fn) fn=dlsym(RTLD_NEXT,"__xstat"); int r=fn(ver,path,st); fprintf(stderr,"XSTAT %s = %d\n",path,r); return r; }
int __lxstat(int ver, const char *path, struct stat *st) { static int (*fn)(int,const char*,struct stat*); if (!fn) fn=dlsym(RTLD_NEXT,"__lxstat"); int r=fn(ver,path,st); fprintf(stderr,"LXSTAT %s = %d\n",path,r); return r; }
int access(const char *path, int mode) { static int (*fn)(const char*,int); if (!fn) fn=dlsym(RTLD_NEXT,"access"); int r=fn(path,mode); fprintf(stderr,"ACCESS %s = %d\n",path,r); return r; }
DIR *opendir(const char *path) { static DIR *(*fn)(const char*); if (!fn) fn=dlsym(RTLD_NEXT,"opendir"); DIR *r=fn(path); fprintf(stderr,"OPENDIR %s = %p\n",path,r); return r; }
ssize_t readlink(const char *path, char *buf, size_t size) { static ssize_t (*fn)(const char*,char*,size_t); if (!fn) fn=dlsym(RTLD_NEXT,"readlink"); ssize_t r=fn(path,buf,size); fprintf(stderr,"READLINK %s = %ld %.*s\n",path,(long)r,(int)(r>0?r:0),buf); return r; }
char *strstr(const char *haystack, const char *needle) { static char *(*fn)(const char*,const char*); if (!fn) fn=dlsym(RTLD_NEXT,"strstr"); char *r=fn(haystack,needle); if (!r && strcmp(needle,"renderD")==0) { if (!haystack[0]) return "renderD0"; if (fn(haystack,"card0")) return (char *)haystack+9; } return r; }
FILE *fopen(const char *path, const char *mode) { static FILE *(*fn)(const char*,const char*); if (!fn) fn=dlsym(RTLD_NEXT,"fopen"); const char *actual=path; if (strstr(path,"/sys/devices/")) { if (strcmp(path+strlen(path)-7,"/vendor")==0 || strcmp(path+strlen(path)-17,"/subsystem_vendor")==0) actual="/tmp/mt-research/fake-vendor"; else if (strcmp(path+strlen(path)-7,"/device")==0 || strcmp(path+strlen(path)-17,"/subsystem_device")==0) actual="/tmp/mt-research/fake-device"; else if (strcmp(path+strlen(path)-7,"/uevent")==0) actual="/tmp/mt-research/fake-uevent"; else if (strcmp(path+strlen(path)-9,"/revision")==0) actual="/tmp/mt-research/fake-revision"; } FILE *r=fn(actual,mode); fprintf(stderr,"FOPEN %s [%s] = %p\n",path,actual,r); return r; }
int open(const char *path, int flags, ...) { mode_t mode = 0; va_list ap; va_start(ap, flags); if (flags & O_CREAT) mode = va_arg(ap, mode_t); va_end(ap); return log_open(path, flags, mode, 0); }
int open64(const char *path, int flags, ...) { mode_t mode = 0; va_list ap; va_start(ap, flags); if (flags & O_CREAT) mode = va_arg(ap, mode_t); va_end(ap); return log_open(path, flags, mode, 1); }
int openat(int dirfd, const char *path, int flags, ...) { mode_t mode = 0; va_list ap; va_start(ap, flags); if (flags & O_CREAT) mode = va_arg(ap, mode_t); va_end(ap); return log_open(path, flags, mode, 2); }
int muDeviceGet(int *dev, int ordinal) { static int (*fn)(int *,int); if (!fn) fn=dlsym(RTLD_NEXT,"muDeviceGet"); int r=fn(dev,ordinal); fprintf(stderr,"REAL muDeviceGet = %d dev %d\n",r,*dev); return r; }
int muDeviceGetCount(int *count) { static int (*fn)(int *); if (!fn) fn=dlsym(RTLD_NEXT,"muDeviceGetCount"); int r=fn(count); fprintf(stderr,"REAL muDeviceGetCount = %d count %d (masked to one)\n",r,*count); *count=1; return 0; }
int muInit(unsigned int flags) { static int (*fn)(unsigned int); if (!fn) fn=dlsym(RTLD_NEXT,"muInit"); int r=fn(flags); fprintf(stderr,"REAL muInit = %d (masked for probe)\n",r); return 0; }
int muCtxSynchronize(uint64_t ctx) { fprintf(stderr,"SPOOF muCtxSynchronize(%#lx) = 0\n",ctx); return 0; }
struct bo_track { uint32_t handle; uint64_t size; void *host; };
struct va_track { uint64_t va, size; void *host; uint32_t bo_handle, pad; };
static struct bo_track bo_tracks[4096];
static struct va_track va_tracks[4096];
static int bo_count, va_count;
static uint32_t next_bo = 1;
static struct bo_track *bo_find(uint32_t handle);
static struct bo_track *bo_add(uint32_t handle, uint64_t size, void *host) {
  struct bo_track *bo = bo_find(handle);
  if (bo) { bo->size = size; bo->host = host; return bo; }
  if (bo_count >= 4096) return NULL;
  bo_tracks[bo_count] = (struct bo_track){handle, size, host};
  return &bo_tracks[bo_count++];
}
static void *fake_allocs[8192];
static size_t fake_alloc_sizes[8192];
static int fake_alloc_count;
static void *sem_hosts[64];
static int sem_count, expect_sem_map;
static struct bo_track *bo_find(uint32_t handle) {
  for (int i = 0; i < bo_count; i++) if (bo_tracks[i].handle == handle) return &bo_tracks[i];
  return NULL;
}
static void *va_host(uint64_t va) {
  for (int i = va_count - 1; i >= 0; i--) if (va >= va_tracks[i].va && va < va_tracks[i].va + va_tracks[i].size)
    return (char *)va_tracks[i].host + (va - va_tracks[i].va);
  return NULL;
}
static const char *capture_path(const char *name) {
  static char path[512];
  const char *dir = getenv("MT_CAPTURE_DIR");
  if (dir) snprintf(path, sizeof(path), "%s/%s", dir, name); else snprintf(path, sizeof(path), "/tmp/mt-research/%s", name);
  return path;
}
static void dump_file(const char *path, const void *data, size_t size) {
  int fd = open(path, O_WRONLY|O_CREAT|O_TRUNC, 0644);
  if (fd >= 0) { if (write(fd, data, size) != (ssize_t)size) fprintf(stderr, "short capture %s\n", path); close(fd); }
}
static void dump_sparse_file(const char *path, const unsigned char *data, size_t size) {
  int fd = open(path, O_WRONLY|O_CREAT|O_TRUNC, 0644);
  if (fd < 0) return;
  uint64_t header[2] = {size, 0};
  if (write(fd, header, sizeof(header)) != sizeof(header)) fprintf(stderr, "short sparse capture %s\n", path);
  for (size_t off = 0; off < size; off += 4096) {
    size_t len = size - off < 4096 ? size - off : 4096, nonzero = 0;
    for (size_t j = 0; j < len; j++) if (data[off+j]) { nonzero = 1; break; }
    if (nonzero) {
      uint32_t entry[2] = {(uint32_t)off, (uint32_t)len};
      if (write(fd, entry, sizeof(entry)) != sizeof(entry) || write(fd, data+off, len) != (ssize_t)len) fprintf(stderr, "short sparse capture %s\n", path);
      header[1]++;
    }
  }
  lseek(fd, 0, SEEK_SET);
  if (write(fd, header, sizeof(header)) != sizeof(header)) fprintf(stderr, "short sparse header %s\n", path);
  close(fd);
}
void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  static void *(*fn)(void *,size_t,int,int,int,off_t);
  if (!fn) fn=dlsym(RTLD_NEXT,"mmap");
  if (fd == 3 || fd == 4) {
    uint32_t handle = (uint64_t)offset >> 32;
    if (!handle) handle = 1;
    struct bo_track *bo = bo_find(handle);
    if (bo && bo->host && bo->size >= length) return bo->host;
    void *p = aligned_alloc(4096, (length + 4095) & ~4095UL);
    if (p) memset(p, 0, length);
    if (bo) { bo->host = p; if (bo->size < length) bo->size = length; }
    for (int i = 0; i < va_count; i++) if (va_tracks[i].bo_handle == handle && !va_tracks[i].host) va_tracks[i].host = p;
    if (p && fake_alloc_count < 8192) { fake_allocs[fake_alloc_count] = p; fake_alloc_sizes[fake_alloc_count] = length; fake_alloc_count++; }
    if (p && expect_sem_map) { if (sem_count < 64) sem_hosts[sem_count++] = p; expect_sem_map = 0; fprintf(stderr, "SEM HOST %p\n", p); }
    fprintf(stderr, "FAKE MMAP handle=%u host=%p length=%#lx\n", handle, p, length);
    return p;
  }
  return fn(addr,length,prot,flags,fd,offset);
}
int munmap(void *addr, size_t length) {
  static int (*fn)(void *, size_t);
  if (!fn) fn = dlsym(RTLD_NEXT, "munmap");
  for (int i = 0; i < fake_alloc_count; i++) if (fake_allocs[i] == addr) return 0;
  return fn(addr, length);
}
int ioctl(int fd, unsigned long req, ...) {
  static int (*real_ioctl)(int, unsigned long, ...);
  if (!real_ioctl) real_ioctl = dlsym(RTLD_NEXT, "ioctl");
  va_list ap; va_start(ap, req); void *arg = va_arg(ap, void *); va_end(ap);
  int ret = real_ioctl(fd, req, arg);
  if (req == 0xc04064a0 || req == 0x40086409) ret = 0;
  if (req == 0xc020644f && arg) {
    struct mtgpu_args_s { uint32_t type, cmd; uint64_t checksum; uint32_t size, pad; uint64_t data; } *a = arg;
    fprintf(stderr, "MTGPU type %u cmd %u size %u data %p\n", a->type, a->cmd, a->size, (void *)a->data);
    void *out = (void *)a->data;
    if (a->type == 0 && a->cmd == 0) { memset(out, 0, a->size); strcpy((char *)out + 8, "MUSA_Driver_Linux_WS musadriver 1.0@0"); strcpy((char *)out + 40, "c66cab6"); }
    if (a->type == 0 && a->cmd == 3) {
      struct versions_s { uint32_t apis, abis, fwifs, pad; uint64_t api, abi, fw; } *v = out;
      v->apis = 64; v->abis = 512; v->fwifs = 256;
      v->api = (uint64_t)api_versions; v->abi = (uint64_t)abi_versions; v->fw = (uint64_t)fw_versions;
    }
    if (a->type == 1 && a->cmd == 2) { uint64_t *m = out; m[0] = m[1] = m[2] = 0x400000000ULL; m[3] = m[4] = 0x10000000000ULL; }
    if (a->type == 1 && a->cmd == 0) *(uint32_t *)out = 16;
    if (a->type == 1 && a->cmd == 1) {
      struct heap_s { uint32_t id; char name[128]; uint64_t base, length; uint32_t log2, multi, mask, align; } *h = (void *)((char *)out + 8);
      static int heap_index;
      static const char *names[] = {"General SVM", "General", "PDS Code and Data", "PDS Indirect State", "USC Code", "Vulkan Capture Replay", "Signals", "Component Control", "FBCDC", "Large FBCDC", "Compute Mission RMW", "Compute Safety RMW", "Texture State", "Visibility Test", "High General SVM", "Video Heap"};
      static const uint64_t bases[] = {0x100000000ULL, 0x8000000000ULL, 0xda00000000ULL, 0xdb00000000ULL, 0xe000000000ULL, 0xe900000000ULL, 0xea00000000ULL, 0xeb00000000ULL, 0xec00000000ULL, 0xec40000000ULL, 0xee00000000ULL, 0xef00000000ULL, 0xf000000000ULL, 0xf200000000ULL, 0x10000000000ULL, 0xff8000000000ULL};
      static const uint64_t lengths[] = {0x7f00000000ULL, 0x59bfff0000ULL, 0xfffff000ULL, 0x1000000ULL, 0xfff80000ULL, 0x40000000ULL, 0x10000ULL, 0x100000000ULL, 0x200000ULL, 0x200000ULL, 0x40000000ULL, 0x40000000ULL, 0x100000000ULL, 0x200000ULL, 0xfe7600000000ULL, 0x8000000000ULL};
      static const uint32_t ids[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
      int i = heap_index++ % 16;
      memset(h, 0, sizeof(*h)); h->id = ids[i]; h->length = lengths[i]; h->log2 = 12; h->multi = 0; h->mask = 0; h->align = 12;
      strcpy(h->name, names[i]); h->base = bases[i];
      if (ids[i] == 14) { h->log2 = 1; h->multi = 282624; h->mask = 0x15; }

    }
    if (a->type == 2 && a->cmd == 0) {
      uint32_t handle = next_bo++;
      *(uint32_t *)((char *)out + 96) = handle;
      struct bo_track *bo = bo_add(handle, *(uint64_t *)out, NULL);
      fprintf(stderr, "BO ALLOC handle=%u size=%#lx\n", handle, *(uint64_t *)out);
    }
    if (a->type == 2 && a->cmd == 1) {
      uint32_t handle = next_bo++;
      *(uint32_t *)((char *)out + 24) = handle;
      bo_add(handle, *(uint64_t *)((char *)out + 8), *(void **)out);
    }
    if (a->type == 2 && a->cmd == 2) {
      uint32_t handle = *(uint32_t *)out;
      *(uint64_t *)((char *)out + 8) = (uint64_t)handle << 32;
    }
    if (a->type == 3 && a->cmd == 2) {
      uint64_t va = *(uint64_t *)((char *)out + 8), size = *(uint64_t *)((char *)out + 32);
      uint32_t handle = *(uint32_t *)((char *)out + 24);
      struct bo_track *bo = bo_find(handle);
      if (va_count < 4096) va_tracks[va_count++] = (struct va_track){va, size, bo ? bo->host : NULL, handle, 0};
      fprintf(stderr, "VM MAP va=%#lx size=%#lx handle=%u host=%p\n", va, size, handle, bo ? bo->host : NULL);
    }
    if (a->type == 5 && a->cmd == 10) {
      fprintf(stderr, "JOB CONTEXT CREATE type=%u flags=%#x priority=%u vm=%#lx\n", *(uint32_t *)out, *(uint32_t *)((char *)out + 4), *(uint32_t *)((char *)out + 8), *(uint64_t *)((char *)out + 16));
    }
    if (a->type == 4 && a->cmd == 5) {
      uint64_t *s = (uint64_t *)((char *)out + 8);
      int idx = sem_count + expect_sem_map;
      uint32_t bo_handle = next_bo++, shadow_handle = next_bo++;
      bo_add(bo_handle, 0x1000, NULL); bo_add(shadow_handle, 0x1000, NULL);
      s[0] = idx + 1; s[1] = bo_handle; s[2] = shadow_handle; s[3] = 0; s[4] = 0xe1c0408000ULL + idx * 0x1000;
      expect_sem_map = 2;
    }
    if (a->type == 5 && a->cmd == 12) {
      uint64_t submission_va = *(uint64_t *)((char *)out + 64);
      uint32_t submission_size = *(uint32_t *)((char *)out + 72);
      void *submission = va_host(submission_va);
      fprintf(stderr, "JOB SUBMIT va=%#lx size=%#x host=%p flags=%#x\n", submission_va, submission_size, submission, *(uint32_t *)((char *)out + 76));

      for (int i = 0; i < sem_count; i++) *(uint64_t *)sem_hosts[i] = 0x100000000ULL;
      if (!getenv("MT_UMD")) alarm(3);
      if (submission) dump_file(capture_path("real-submission.bin"), submission, submission_size);
      for (int i = 0; i < fake_alloc_count; i++) {
        int nonzero = 0;
        for (size_t j = 0; j < fake_alloc_sizes[i]; j++) if (((unsigned char *)fake_allocs[i])[j]) { nonzero = 1; break; }
        if (nonzero) {
          char path[256]; snprintf(path, sizeof(path), "real-alloc-%d.bin", i);
          fprintf(stderr, "NONZERO ALLOC alloc=%d host=%p allocsize=%#lx\n", i, fake_allocs[i], fake_alloc_sizes[i]);
          dump_file(capture_path(path), fake_allocs[i], fake_alloc_sizes[i]);
        }
      }
      for (int i = 0; i < va_count; i++) if (va_tracks[i].host && va_tracks[i].size <= 0x1000000) {
        char path[256]; snprintf(path, sizeof(path), "real-map-%lx.sparse", va_tracks[i].va);
        dump_sparse_file(capture_path(path), va_tracks[i].host, va_tracks[i].size);
      }
      uint32_t done = 1;
      dump_file(capture_path("real-capture-done"), &done, sizeof(done));
    }
    if (a->type == 3 && a->cmd == 0) *(uint64_t *)((char *)out + 8) = 1;
    if (a->type == 4 && a->cmd == 0) {
      uint64_t *s = (uint64_t *)((char *)out + 8);
      int idx = sem_count + expect_sem_map;
      s[0] = idx + 1; s[1] = 1; s[2] = 1; s[3] = 0; s[4] = 0xe1c0408000ULL + idx * 0x1000;
      expect_sem_map = 1;
    }
    if (a->type == 1 && a->cmd == 4) {
      struct dev_s { uint32_t id; char name[48]; uint32_t status, clock, memclock, memmax, cores, mpx, soc_clock; char uuid[16]; uint16_t subvendor, subsystem; uint32_t llcmax, llcsize; uint8_t igpu, doorbells; uint32_t channels; uint16_t flags; } *d = out;
      memset(d, 0, sizeof(*d)); d->id = 0x201; strcpy(d->name, "MTT S80"); d->status = 1; d->clock = 1600000000; d->memclock = 1600000000; d->memmax = 1600000000; d->soc_clock = 25000000; d->cores = 8; d->mpx = getenv("MT_MPX") ? strtoul(getenv("MT_MPX"), NULL, 0) : 1; d->channels = 0xff;
      d->subvendor = 0x1ed5; d->subsystem = 0x0201; d->llcmax = 0x2000000; d->llcsize = 0x1000000;
      static const uint8_t uuid[16] = {0x4d,0x54,0x54,0x20,0x53,0x38,0x30,0x20,0x51,0x55,0x59,0x55,0x41,0x4e,0x31,0x00};
      memcpy(d->uuid, uuid, 16);
    }
    if (a->type == 1 && a->cmd == 5) {
      struct pci_s { uint32_t domain, bus, device, function, speed, width; int32_t numa; uint8_t all_pci, all_sys, all_user, no_snoop; } *p = out;
      memset(p, 0, sizeof(*p)); p->bus = 2; p->speed = 5; p->width = 16; p->numa = 0;
      p->all_pci = p->all_sys = p->all_user = 1; p->no_snoop = 0;
    }
    if (a->type == 1 && a->cmd == 6) { memset(out, 0, 12); ((uint8_t *)out)[4] = 1; }
    if (a->type == 1 && a->cmd == 7) *(uint64_t *)out = 4;
    if (a->type == 1 && a->cmd == 10) { memset(out, 0, 8); ((uint8_t *)out)[0] = 1; ((uint8_t *)out)[1] = 1; ((uint8_t *)out)[2] = 1; }
    ret = 0;
  }
  if (req == 0xc0406400 && arg) {
    struct drm_version_s { int major, minor, patch; size_t name_len; char *name; size_t date_len; char *date; size_t desc_len; char *desc; } *v = arg;
    char *name = strdup("mtgpu");
    v->major = 1; v->minor = 0; v->patch = 0;
    v->name = name; v->name_len = strlen(name);
    ret = 0;
  }
  fprintf(stderr, "IOCTL fd %d req %#lx = %d\n", fd, req, ret);
  return ret;
}
