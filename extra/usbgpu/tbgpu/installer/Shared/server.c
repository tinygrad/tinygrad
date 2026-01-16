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
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <mach/mach.h>

// Protocol

enum {
  CMD_MAP_BAR = 1,        // map PCI BAR, returns size
  CMD_MAP_SYSMEM_FD = 2,  // alloc DMA memory, returns fd via SCM_RIGHTS
  CMD_CFG_READ = 3,       // read PCI config space
  CMD_CFG_WRITE = 4,      // write PCI config space
  CMD_RESET = 5,          // reset device
  CMD_MMIO_READ = 6,      // bulk read from BAR
  CMD_MMIO_WRITE = 7,     // bulk write to BAR
  RESP_OK = 0, RESP_ERR = 1,
};

typedef struct { uint8_t cmd, bar; uint64_t offset, size, value; } __attribute__((packed)) request_t;
typedef struct { uint8_t status; uint64_t value, addr; } __attribute__((packed)) response_t;

#define RESP(call) { resp.status = (call) ? RESP_ERR : RESP_OK; break; }
#define RESP_VAL(call, v) { int _r = (call); resp.status = _r ? RESP_ERR : RESP_OK; if (!_r) resp.value = (v); break; }
#define RESP_ERR_BREAK() { resp.status = RESP_ERR; break; }
#define RESP_SEND_DATA(sz, data) { resp.status = RESP_OK; resp.value = (sz); send_response(fd, &resp, -1); send(fd, data, sz, 0); send_resp = 0; break; }
#define NO_RESP() { send_resp = 0; break; }

// Constants and state

#define BULK_BUF_SIZE (16 << 20)
#define MAX_BARS 6
#define MAX_SYSMEM 128

static uint8_t g_bulk_buf[BULK_BUF_SIZE];
static io_connect_t g_conn = IO_OBJECT_NULL;
static int g_client_active = 0;

static struct { mach_vm_address_t addr; mach_vm_size_t size; int mapped; } g_bars[MAX_BARS];
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
  volatile uint8_t *d1 = (volatile uint8_t*)(d + len / 4);
  volatile uint8_t *s1 = (volatile uint8_t*)(s + len / 4);
  for (size_t i = 0; i < len % 4; i++) d1[i] = s1[i];
}

static int send_response(int fd, response_t *resp, int send_fd) {
  struct iovec iov = {.iov_base = resp, .iov_len = sizeof(*resp)};
  struct msghdr msg = {.msg_iov = &iov, .msg_iovlen = 1};
  char cmsgbuf[CMSG_SPACE(sizeof(int))];
  if (send_fd >= 0) {
    msg.msg_control = cmsgbuf;
    msg.msg_controllen = sizeof(cmsgbuf);
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    *cmsg = (struct cmsghdr){.cmsg_level = SOL_SOCKET, .cmsg_type = SCM_RIGHTS, .cmsg_len = CMSG_LEN(sizeof(int))};
    memcpy(CMSG_DATA(cmsg), &send_fd, sizeof(int));
  }
  return sendmsg(fd, &msg, 0) > 0 ? 0 : -1;
}

static void send_error_msg(int fd, const char *msg) {
  response_t resp = {.status = RESP_ERR, .value = strlen(msg)};
  send_response(fd, &resp, -1);
  send(fd, msg, strlen(msg), 0);
}

// Driver interface

static io_connect_t open_tinygpu(void) {
  io_service_t svc = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceNameMatching("tinygpu"));
  if (!svc) return IO_OBJECT_NULL;
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
  if (g_bars[bar].mapped) {
    resp->addr = g_bars[bar].addr;
    resp->value = g_bars[bar].size;
    return 0;
  }
  mach_vm_address_t addr = 0;
  mach_vm_size_t size = 0;
  if (IOConnectMapMemory64(g_conn, bar, mach_task_self(), &addr, &size, kIOMapAnywhere) != KERN_SUCCESS) return -1;
  g_bars[bar] = (typeof(g_bars[bar])){.addr = addr, .size = size, .mapped = 1};
  resp->addr = addr;
  resp->value = size;
  printf("BAR%u mapped: 0x%llx size 0x%llx\n", bar, addr, size);
  return 0;
}

static int map_sysmem_fd(uint64_t size, response_t *resp, int *out_fd) {
  if (g_sysmem_count >= MAX_SYSMEM) return -1;
  int idx = g_sysmem_count;

  // page-align, ensure >= 4097 for IOMemoryDescriptor in dext
  size_t page_sz = getpagesize();
  size_t alloc_sz = ((size + page_sz - 1) & ~(page_sz - 1));
  if (alloc_sz < 4097) alloc_sz = 16 << 10;

  // create POSIX shared memory
  char shm_name[32];
  snprintf(shm_name, sizeof(shm_name), "/tinygpu_%d", idx);
  shm_unlink(shm_name);
  int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0600);
  if (fd < 0) { perror("shm_open"); return -1; }
  if (ftruncate(fd, alloc_sz) < 0) { perror("ftruncate"); close(fd); shm_unlink(shm_name); return -1; }

  void *ptr = mmap(NULL, alloc_sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) { perror("mmap"); close(fd); shm_unlink(shm_name); return -1; }

  // call PrepareDMA (selector 3) - writes physical addresses to output buffer
  size_t out_sz = 8192;
  void *out_buf = mmap(NULL, out_sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (out_buf == MAP_FAILED) { perror("mmap out"); munmap(ptr, alloc_sz); close(fd); shm_unlink(shm_name); return -1; }
  memset(out_buf, 0, out_sz);

  kern_return_t kr = IOConnectCallStructMethod(g_conn, 3, ptr, alloc_sz, out_buf, &out_sz);
  if (kr != KERN_SUCCESS) {
    printf("PrepareDMA failed: 0x%x\n", kr);
    munmap(out_buf, 8192); munmap(ptr, alloc_sz); close(fd); shm_unlink(shm_name);
    return -1;
  }

  // copy paddrs to start of shared buffer for client to read
  memcpy(ptr, out_buf, out_sz);
  munmap(out_buf, 8192);
  printf("sysmem[%d]: %p size 0x%zx fd=%d phys=0x%llx\n", idx, ptr, alloc_sz, fd, ((uint64_t*)ptr)[0]);

  g_sysmem[idx] = (typeof(g_sysmem[idx])){.addr = (mach_vm_address_t)ptr, .size = alloc_sz, .shm_fd = fd};
  strncpy(g_sysmem[idx].shm_name, shm_name, sizeof(g_sysmem[idx].shm_name));
  g_sysmem_count++;

  *resp = (response_t){.addr = idx, .value = alloc_sz};
  *out_fd = fd;
  return 0;
}

static int validate_bar(uint8_t bar, uint64_t off, uint64_t sz) {
  return (bar < MAX_BARS && g_bars[bar].mapped && off + sz <= g_bars[bar].size && sz <= BULK_BUF_SIZE) ? 0 : -1;
}

static void cleanup(void) {
  for (int i = 0; i < MAX_BARS; i++)
    if (g_bars[i].mapped) { IOConnectUnmapMemory64(g_conn, i, mach_task_self(), g_bars[i].addr); g_bars[i].mapped = 0; }
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

  g_conn = open_tinygpu();
  if (g_conn == IO_OBJECT_NULL) {
    fprintf(stderr, "failed to connect to tinygpu driver\n");
    request_t req; recv(fd, &req, sizeof(req), 0);
    send_error_msg(fd, "Driver not available. Check: System Report > PCI for GPU, System Settings > Privacy & Security.");
    return;
  }

  request_t req;
  response_t resp;
  while (recv(fd, &req, sizeof(req), 0) == sizeof(req)) {
    resp = (response_t){0};
    int send_resp = 1;

    switch (req.cmd) {
      case CMD_MAP_BAR: RESP(map_bar(req.bar, &resp));
      case CMD_MAP_SYSMEM_FD: {
        int shm_fd = -1;
        if (map_sysmem_fd(req.size, &resp, &shm_fd) == 0) send_response(fd, &resp, shm_fd);
        else { resp.status = RESP_ERR; send_response(fd, &resp, -1); }
        send_resp = 0;
        break;
      }
      case CMD_CFG_READ: {
        uint64_t in[2] = {req.offset, req.size}, out;
        RESP_VAL(dext_rpc(0, in, 2, &out), (resp.value = out));
      }
      case CMD_CFG_WRITE: {
        uint64_t in[3] = {req.offset, req.size, req.value};
        RESP(dext_rpc(1, in, 3, NULL));
      }
      case CMD_RESET: RESP(dext_rpc(2, NULL, 0, NULL));
      case CMD_MMIO_READ:
        if (validate_bar(req.bar, req.offset, req.size)) RESP_ERR_BREAK();
        mmio_copy(g_bulk_buf, (void*)(g_bars[req.bar].addr + req.offset), req.size);
        RESP_SEND_DATA(req.size, g_bulk_buf);
      case CMD_MMIO_WRITE:
        recvall(fd, g_bulk_buf, req.size);
        if (validate_bar(req.bar, req.offset, req.size)) NO_RESP();
        mmio_copy((void*)(g_bars[req.bar].addr + req.offset), g_bulk_buf, req.size);
        NO_RESP();
      default: RESP_ERR_BREAK();
    }
    if (send_resp) send_response(fd, &resp, -1);
  }

  printf("client disconnected\n");
  cleanup();
}

int run_server(const char *sock_path) {
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
