/* Constants from nvidia-oot/include/linux/nvmap.h (kernel-internal header).
   The uapi header (uapi/linux/nvmap.h) doesn't export these, but they're needed
   for nvmap_alloc_handle.heap_mask and .flags fields. */
#define NVMAP_HEAP_IOVMM            (1 << 30)
#define NVMAP_HANDLE_UNCACHEABLE    (0 << 0)
#define NVMAP_HANDLE_WRITE_COMBINE  (1 << 0)
#define NVMAP_HANDLE_INNER_CACHEABLE (2 << 0)
#define NVMAP_HANDLE_CACHEABLE      (3 << 0)
