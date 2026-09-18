#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <dispatch/dispatch.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/IOMessage.h>
#include <mach/mach.h>

// Protocol

enum {
  CMD_PROBE = 0,          // probe devices, returns count
  CMD_MAP_BAR = 1,        // map PCI BAR, returns size
  CMD_MAP_SYSMEM_FD = 2,  // alloc DMA memory, returns fd via SCM_RIGHTS
  CMD_CFG_READ = 3,       // read PCI config space
  CMD_CFG_WRITE = 4,      // write PCI config space
  CMD_RESET = 5,          // reset device
  CMD_MMIO_READ = 6,      // bulk read from BAR
  CMD_MMIO_WRITE = 7,     // bulk write to BAR
  CMD_MAP_SYSMEM = 8,     // map system memory
  CMD_SYSMEM_READ = 9,    // bulk read from system memory
  CMD_SYSMEM_WRITE = 10,  // bulk write to system memory
  CMD_RESIZE_BAR = 11,    // resize bar (noop)
  RESP_OK = 0, RESP_ERR = 1,
};

typedef struct { uint8_t cmd; uint32_t dev_id, bar; uint64_t arg0, arg1, arg2; } __attribute__((packed)) request_t;
typedef struct { uint8_t status; uint64_t resp0, resp1; } __attribute__((packed)) response_t;

// Constants and state

#define BULK_BUF_SIZE (64 << 20)
#define MAX_BARS 6
#define MAX_SYSMEM 128

static uint8_t g_bulk_buf[BULK_BUF_SIZE];
static io_connect_t g_conn = IO_OBJECT_NULL;
static int g_client_active = 0;

static struct { mach_vm_address_t addr; mach_vm_size_t size; } g_bars[MAX_BARS];
static struct { mach_vm_address_t addr; mach_vm_size_t size; int shm_fd; char shm_name[32]; } g_sysmem[MAX_SYSMEM];
static int g_sysmem_count = 0;

// Utilities

static void recvall(int fd, void *buf, size_t len) {
  for (size_t off = 0; off < len; ) {
    ssize_t r = recv(fd, (uint8_t*)buf + off, len - off, 0);
    if (r <= 0) break;
    off += r;
  }
}

// MMIO requires 32-bit aligned volatile accesses
static void mmio_copy(void *dst, void *src, size_t len) {
  volatile uint32_t *d = dst, *s = src;
  for (size_t i = 0; i < len / 4; i++) d[i] = s[i];
  for (size_t i = len & ~3; i < len; i++) ((volatile uint8_t*)dst)[i] = ((volatile uint8_t*)src)[i];
}

static int send_response(int fd, response_t *resp, int send_fd) {
  char cmsgbuf[CMSG_SPACE(sizeof(int))];
  struct iovec iov = {resp, sizeof(*resp)};
  struct msghdr msg = {.msg_iov = &iov, .msg_iovlen = 1};

  if (send_fd >= 0) {
    msg.msg_control = cmsgbuf;
    msg.msg_controllen = sizeof(cmsgbuf);
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    *cmsg = (struct cmsghdr){.cmsg_level = SOL_SOCKET, .cmsg_type = SCM_RIGHTS, .cmsg_len = CMSG_LEN(sizeof(int))};
    memcpy(CMSG_DATA(cmsg), &send_fd, sizeof(int));
  }
  return sendmsg(fd, &msg, 0) > 0 ? 0 : -1;
}

static void send_error(int fd, const char *msg) {
  response_t resp = {.status = RESP_ERR, .resp0 = strlen(msg)};
  send_response(fd, &resp, -1);
  send(fd, msg, strlen(msg), 0);
}

// Driver interface

static void on_disconnect(void *refcon, io_service_t svc, uint32_t msg, void *arg) {
  if (msg == kIOMessageServiceIsTerminated) _exit(0);
}

// Which of the Mac's tinygpu services this server drives: `server <sock> --device <i>` picks the
// i-th in ascending registry-entry-id order (the order the cards were matched in). Default 0.
static uint32_t g_device_index = 0;
#define MAX_DEVICES 8
typedef struct { io_service_t svc; uint64_t entry_id; } tinygpu_service_t;

// Every tinygpu service, sorted by registry entry id; the caller releases them.
static size_t list_tinygpu_services(tinygpu_service_t *out, size_t max) {
  io_iterator_t iter = IO_OBJECT_NULL;
  if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceNameMatching("tinygpu"), &iter) != KERN_SUCCESS) return 0;
  size_t n = 0;
  io_service_t svc;
  while ((svc = IOIteratorNext(iter)) != IO_OBJECT_NULL) {
    uint64_t entry_id = 0;
    if (n < max && IORegistryEntryGetRegistryEntryID(svc, &entry_id) == KERN_SUCCESS) out[n++] = (tinygpu_service_t){svc, entry_id};
    else IOObjectRelease(svc);
  }
  IOObjectRelease(iter);
  for (size_t i = 1; i < n; i++)
    for (size_t j = i; j > 0 && out[j - 1].entry_id > out[j].entry_id; j--) { tinygpu_service_t t = out[j]; out[j] = out[j - 1]; out[j - 1] = t; }
  return n;
}

// A 32-bit property of the service's provider (the IOPCIDevice): vendor-id, device-id. 0 when absent.
static uint32_t provider_u32(io_service_t svc, const char *key) {
  io_registry_entry_t parent = IO_OBJECT_NULL;
  if (IORegistryEntryGetParentEntry(svc, kIOServicePlane, &parent) != KERN_SUCCESS) return 0;
  CFStringRef cfkey = CFStringCreateWithCString(NULL, key, kCFStringEncodingUTF8);
  CFTypeRef value = IORegistryEntryCreateCFProperty(parent, cfkey, kCFAllocatorDefault, 0);
  CFRelease(cfkey);
  IOObjectRelease(parent);
  uint32_t result = 0;
  if (value) {
    if (CFGetTypeID(value) == CFDataGetTypeID()) {
      uint8_t buf[4] = {0};
      CFIndex len = CFDataGetLength((CFDataRef)value);
      CFDataGetBytes((CFDataRef)value, CFRangeMake(0, len < 4 ? len : 4), buf);
      result = (uint32_t)buf[0] | (uint32_t)buf[1] << 8 | (uint32_t)buf[2] << 16 | (uint32_t)buf[3] << 24;
    } else if (CFGetTypeID(value) == CFNumberGetTypeID()) {
      int64_t num = 0;
      CFNumberGetValue((CFNumberRef)value, kCFNumberSInt64Type, &num);
      result = (uint32_t)num;
    }
    CFRelease(value);
  }
  return result;
}

// CMD_PROBE: one line per tinygpu service in --device order, "<vendor>:<device>:<registry entry id>:<index>"
// (hex, hex, hex, decimal) — the shape RemotePCIDevice.scan() reads (data length in resp0, then the text;
// the last ':' field is the dev_id). The request's payload (arg1 bytes) is consumed and ignored.
static void probe_devices(int fd, const request_t *req) {
  for (uint64_t left = req->arg1; left > 0; ) {
    uint8_t skip[256];
    size_t take = left < sizeof(skip) ? (size_t)left : sizeof(skip);
    recvall(fd, skip, take);
    left -= take;
  }
  tinygpu_service_t services[MAX_DEVICES];
  size_t n = list_tinygpu_services(services, MAX_DEVICES);
  char text[MAX_DEVICES * 48];
  size_t len = 0;
  for (size_t i = 0; i < n; i++) {
    uint32_t vendor = provider_u32(services[i].svc, "vendor-id") & 0xffff, device = provider_u32(services[i].svc, "device-id") & 0xffff;
    len += (size_t)snprintf(text + len, sizeof(text) - len, "%s%04x:%04x:%llx:%zu", i ? "\n" : "", vendor, device,
                            (unsigned long long)services[i].entry_id, i);
    IOObjectRelease(services[i].svc);
  }
  response_t resp = {.status = RESP_OK, .resp0 = len, .resp1 = n};
  send_response(fd, &resp, -1);
  if (len) send(fd, text, len, 0);
}

static io_connect_t open_tinygpu(void) {
  static io_object_t notif;
  tinygpu_service_t services[MAX_DEVICES];
  size_t n = list_tinygpu_services(services, MAX_DEVICES);
  io_service_t svc = IO_OBJECT_NULL;
  for (size_t i = 0; i < n; i++) {
    if (i == g_device_index) svc = services[i].svc; else IOObjectRelease(services[i].svc);
  }
  if (!svc) return IO_OBJECT_NULL;

  if (!notif) {
    IONotificationPortRef port = IONotificationPortCreate(kIOMainPortDefault);
    IONotificationPortSetDispatchQueue(port, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0));
    IOServiceAddInterestNotification(port, svc, kIOGeneralInterest, on_disconnect, NULL, &notif);
  }

  io_connect_t conn;
  kern_return_t kr = IOServiceOpen(svc, mach_task_self(), 0, &conn);
  IOObjectRelease(svc);
  return kr == KERN_SUCCESS ? conn : IO_OBJECT_NULL;
}

static int dext_rpc(uint32_t sel, uint64_t *in, uint32_t in_cnt, uint64_t *out_val) {
  uint64_t out[2];
  uint32_t out_cnt = 2;
  if (IOConnectCallMethod(g_conn, sel, in, in_cnt, NULL, 0, out, &out_cnt, NULL, NULL) != KERN_SUCCESS) return -1;
  if (out_val) *out_val = out[0];
  return 0;
}

static int map_bar(uint32_t bar, response_t *resp) {
  if (bar >= MAX_BARS) return -1;
  if (!g_bars[bar].addr && IOConnectMapMemory64(g_conn, bar, mach_task_self(), &g_bars[bar].addr, &g_bars[bar].size, kIOMapAnywhere)) return -1;
  resp->resp0 = g_bars[bar].addr;
  resp->resp1 = g_bars[bar].size;
  return 0;
}

static int map_sysmem_fd(uint64_t size, int contiguous, response_t *resp, int *out_fd) {
  if (g_sysmem_count >= MAX_SYSMEM) return -1;
  int idx = g_sysmem_count;
  int fd = -1;
  void *ptr = MAP_FAILED;
  char shm_name[32];

  // page-align, min 16KB for IOMemoryDescriptor
  size_t alloc_sz = (size + 0xfff) & ~0xfff;
  if (alloc_sz < 0x4000) alloc_sz = 0x4000;

  snprintf(shm_name, sizeof(shm_name), "/tinygpu_%d", idx);
  shm_unlink(shm_name);
  if ((fd = shm_open(shm_name, O_CREAT | O_RDWR, 0600)) < 0) goto fail;
  if (ftruncate(fd, alloc_sz) < 0) goto fail;
  if ((ptr = mmap(NULL, alloc_sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)) == MAP_FAILED) goto fail;

  // PrepareDMA writes physical addresses to output buffer, copy to shared mem
  uint8_t paddr_buf[8192] = {0};
  size_t out_sz = sizeof(paddr_buf);
  if (IOConnectCallStructMethod(g_conn, 3, ptr, alloc_sz, paddr_buf, &out_sz) != KERN_SUCCESS) goto fail;
  memcpy(ptr, paddr_buf, out_sz);

  g_sysmem[idx] = (typeof(g_sysmem[idx])){.addr = (mach_vm_address_t)ptr, .size = alloc_sz, .shm_fd = fd};
  strncpy(g_sysmem[idx].shm_name, shm_name, sizeof(g_sysmem[idx].shm_name));
  g_sysmem_count++;

  *resp = (response_t){.resp0 = alloc_sz, .resp1 = idx};
  *out_fd = fd;
  return 0;

fail:
  if (ptr != MAP_FAILED) munmap(ptr, alloc_sz);
  if (fd >= 0) close(fd);
  shm_unlink(shm_name);
  return -1;
}

static int validate_bar(uint8_t bar, uint64_t off, uint64_t sz) {
  return (bar < MAX_BARS && g_bars[bar].addr && off + sz <= g_bars[bar].size && sz <= BULK_BUF_SIZE) ? 0 : -1;
}

static void cleanup(void) {
  for (int i = 0; i < MAX_BARS; i++)
    if (g_bars[i].addr) { IOConnectUnmapMemory64(g_conn, i, mach_task_self(), g_bars[i].addr); g_bars[i].addr = 0; }

  for (int i = 0; i < g_sysmem_count; i++) {
    munmap((void*)g_sysmem[i].addr, g_sysmem[i].size);
    close(g_sysmem[i].shm_fd);
    shm_unlink(g_sysmem[i].shm_name);
  }

  g_sysmem_count = 0;
  if (g_conn != IO_OBJECT_NULL) { IOServiceClose(g_conn); g_conn = IO_OBJECT_NULL; }
}

static void handle_client(int fd) {
  int bufsize = BULK_BUF_SIZE;
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));
  printf("client connected\n");

  // The device is opened on the first command that needs it, so a client can PROBE
  // (list the cards) a server whose own card is not there.
  request_t req;
  response_t resp;
  while (recv(fd, &req, sizeof(req), 0) == sizeof(req)) {
    resp = (response_t){0};

    if (req.cmd == CMD_PROBE) { probe_devices(fd, &req); continue; }
    if (g_conn == IO_OBJECT_NULL) {
      g_conn = open_tinygpu();
      if (g_conn == IO_OBJECT_NULL) {
        fprintf(stderr, "failed to connect to tinygpu driver (device %u)\n", g_device_index);
        send_error(fd, "Driver not available. Check: System Report > PCI for GPU, System Settings > Privacy & Security.");
        return;
      }
    }

    switch (req.cmd) {
    case CMD_MAP_BAR:
      resp.status = map_bar(req.bar, &resp) ? 1 : 0;
      break;

    case CMD_MAP_SYSMEM_FD: {
      int shm_fd = -1;
      resp.status = map_sysmem_fd(req.arg0, (int)req.arg1, &resp, &shm_fd) ? 1 : 0;
      send_response(fd, &resp, shm_fd);
      continue;
    }

    case CMD_CFG_READ: {
      uint64_t in[2] = {req.arg0, req.arg1};
      resp.status = dext_rpc(0, in, 2, &resp.resp0) ? 1 : 0;
      break;
    }

    case CMD_CFG_WRITE: {
      uint64_t in[3] = {req.arg0, req.arg1, req.arg2};
      resp.status = dext_rpc(1, in, 3, NULL) ? 1 : 0;
      break;
    }

    case CMD_RESIZE_BAR:
      break;

    case CMD_RESET:
      resp.status = dext_rpc(2, NULL, 0, NULL) ? 1 : 0;
      break;

    case CMD_MMIO_READ:
      if (validate_bar(req.bar, req.arg0, req.arg1)) { resp.status = 1; break; }
      mmio_copy(g_bulk_buf, (void*)(g_bars[req.bar].addr + req.arg0), req.arg1);
      resp.resp0 = req.arg1;
      send_response(fd, &resp, -1);
      send(fd, g_bulk_buf, req.arg1, 0);
      continue;

    case CMD_MMIO_WRITE:
      recvall(fd, g_bulk_buf, req.arg1);
      if (!validate_bar(req.bar, req.arg0, req.arg1))
        mmio_copy((void*)(g_bars[req.bar].addr + req.arg0), g_bulk_buf, req.arg1);
      continue;

    default:
      resp.status = 1;
    }
    send_response(fd, &resp, -1);
  }

  printf("client disconnected\n");
  cleanup();
}

int run_server(const char *sock_path, uint32_t device_index) {
  g_device_index = device_index;
  int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (server_fd < 0) { perror("socket"); return 1; }

  struct sockaddr_un addr = {.sun_family = AF_UNIX};
  strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);
  unlink(sock_path);

  if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); close(server_fd); return 1; }
  if (listen(server_fd, 1) < 0) { perror("listen"); close(server_fd); return 1; }
  printf("listening on %s\n", sock_path);

  while (1) {
    int client_fd = accept(server_fd, NULL, NULL);
    if (client_fd < 0) { if (errno == EINTR) continue; perror("accept"); break; }
    if (g_client_active) { printf("rejected: client already connected\n"); close(client_fd); continue; }
    g_client_active = 1;
    handle_client(client_fd);
    g_client_active = 0;
    close(client_fd);
  }

  close(server_fd);
  unlink(sock_path);
  cleanup();
  return 0;
}
