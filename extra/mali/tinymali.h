#ifndef TINYMALI_H
#define TINYMALI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct tm_context;
struct tm_buffer;
struct tm_kernel;

struct tm_context *tm_create(const char *path);
void tm_destroy(struct tm_context *ctx);
int tm_synchronize(struct tm_context *ctx);
struct tm_buffer *tm_buffer_create(struct tm_context *ctx, uint64_t size);
void tm_buffer_destroy(struct tm_context *ctx, struct tm_buffer *buf);
struct tm_buffer *tm_buffer_offset(struct tm_context *ctx, struct tm_buffer *buf, uint64_t offset);
int tm_buffer_write(struct tm_context *ctx, struct tm_buffer *buf, uint64_t offset, const void *src, uint64_t size);
int tm_buffer_read(struct tm_context *ctx, struct tm_buffer *buf, uint64_t offset, void *dst, uint64_t size);
struct tm_kernel *tm_compile(struct tm_context *ctx, const void *src, uint64_t size);
void tm_kernel_destroy(struct tm_context *ctx, struct tm_kernel *kernel);
int tm_launch(struct tm_context *ctx, struct tm_kernel *kernel,
              struct tm_buffer **buffers, const uint32_t *offsets, const uint32_t *image_info, uint32_t num_buffers,
              void *args, uint32_t args_size,
              const uint32_t grid[3], const uint32_t block[3], uint32_t shared_mem);

#ifdef __cplusplus
}
#endif
#endif
