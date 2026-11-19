/* Native compute shim for tinygrad: Gallium Panfrost over panthor, no OpenCL. */
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tinymali.h"
#include "pipe/p_context.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "pipe/p_defines.h"
#include "util/u_inlines.h"
#include "util/u_sampler.h"
#include "util/blob.h"
#include "util/os_time.h"
#include "compiler/glsl_types.h"
#include "nir/nir_serialize.h"
#include "panfrost/lib/pan_shader.h"
#include "gallium/winsys/panfrost/drm/panfrost_drm_public.h"
#include "util/xmlconfig.h"
#include "util/driconf.h"

struct tm_context {
  struct pipe_screen *screen;
  struct pipe_context *pipe;
};

struct tm_buffer {
  struct pipe_resource *res;
  uint64_t offset;
};

struct tm_kernel {
  void *cso;
  uint32_t num_samplers;
  uint32_t num_images;
};

struct tm_context *tm_create(const char *path) {
  struct tm_context *ctx = calloc(1, sizeof(*ctx));
  if (!ctx) return NULL;
  int fd = open(path, O_RDWR | O_CLOEXEC);
  if (fd < 0) { free(ctx); return NULL; }
  static const driOptionDescription panfrost_driconf[] = {
#include "gallium/drivers/panfrost/driinfo_panfrost.h"
  };
  driOptionCache option_info = {}, option_cache = {};
  driParseOptionInfo(&option_info, panfrost_driconf, ARRAY_SIZE(panfrost_driconf));
  struct pipe_screen_config config = {
    .driver_name_is_inferred = false,
    .options_info = &option_info,
    .options = &option_cache,
  };
  glsl_type_singleton_init_or_ref();
  ctx->screen = panfrost_drm_screen_create(fd, &config);
  close(fd);
  driDestroyOptionCache(&option_cache);
  driDestroyOptionInfo(&option_info);
  if (!ctx->screen) { glsl_type_singleton_decref(); free(ctx); return NULL; }
  ctx->pipe = ctx->screen->context_create(ctx->screen, NULL,
    PIPE_CONTEXT_COMPUTE_ONLY | PIPE_CONTEXT_NO_LOD_BIAS);
  if (!ctx->pipe) {
    ctx->screen->destroy(ctx->screen);
    glsl_type_singleton_decref();
    free(ctx);
    return NULL;
  }
  return ctx;
}

void tm_destroy(struct tm_context *ctx) {
  if (!ctx) return;
  ctx->pipe->destroy(ctx->pipe);
  ctx->screen->destroy(ctx->screen);
  glsl_type_singleton_decref();
  free(ctx);
}

int tm_synchronize(struct tm_context *ctx) {
  struct pipe_fence_handle *fence = NULL;
  ctx->pipe->flush(ctx->pipe, &fence, 0);
  if (!fence) return 0;
  bool ok = ctx->screen->fence_finish(ctx->screen, NULL, fence, OS_TIMEOUT_INFINITE);
  ctx->screen->fence_reference(ctx->screen, &fence, NULL);
  return ok ? 0 : -ETIMEDOUT;
}

struct tm_buffer *tm_buffer_create(struct tm_context *ctx, uint64_t size) {
  if (!size || size > UINT32_MAX) { errno = EINVAL; return NULL; }
  struct tm_buffer *buf = calloc(1, sizeof(*buf));
  if (!buf) return NULL;
  struct pipe_resource tmpl = {
    .target = PIPE_BUFFER,
    .width0 = (uint32_t)size,
    .height0 = 1,
    .depth0 = 1,
    .array_size = 1,
    .bind = PIPE_BIND_GLOBAL | PIPE_BIND_SHADER_BUFFER | PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_SHADER_IMAGE | PIPE_BIND_LINEAR,
    .usage = PIPE_USAGE_DEFAULT,
  };
  buf->res = ctx->screen->resource_create(ctx->screen, &tmpl);
  if (!buf->res) { free(buf); return NULL; }
  return buf;
}

void tm_buffer_destroy(struct tm_context *ctx, struct tm_buffer *buf) {
  if (!buf) return;
  tm_synchronize(ctx);
  pipe_resource_reference(&buf->res, NULL);
  free(buf);
}

struct tm_buffer *tm_buffer_offset(struct tm_context *ctx, struct tm_buffer *buf, uint64_t offset) {
  struct tm_buffer *view = calloc(1, sizeof(*view));
  if (!view) return NULL;
  view->res = buf->res;
  view->offset = buf->offset + offset;
  pipe_resource_reference(&view->res, view->res);
  return view;
}

int tm_buffer_write(struct tm_context *ctx, struct tm_buffer *buf, uint64_t offset, const void *src, uint64_t size) {
  if (!buf || buf->offset + offset + size > buf->res->width0 || offset + size < offset) return -EINVAL;
  pipe_buffer_write(ctx->pipe, buf->res, buf->offset + offset, size, src);
  return 0;
}

int tm_buffer_read(struct tm_context *ctx, struct tm_buffer *buf, uint64_t offset, void *dst, uint64_t size) {
  if (!buf || buf->offset + offset + size > buf->res->width0 || offset + size < offset) return -EINVAL;
  if (tm_synchronize(ctx)) return -EIO;
  pipe_buffer_read(ctx->pipe, buf->res, buf->offset + offset, size, dst);
  return 0;
}

struct tm_kernel *tm_compile(struct tm_context *ctx, const void *src, uint64_t size) {
  struct blob_reader reader;
  blob_reader_init(&reader, src, size);
  const nir_shader_compiler_options *opts = pan_shader_get_compiler_options(10);
  nir_shader *nir = nir_deserialize(NULL, opts, &reader);
  if (!nir) { errno = EINVAL; return NULL; }
  uint32_t num_samplers = nir->info.num_textures;
  uint32_t num_images = nir->info.num_images;
  struct pipe_compute_state state = {
    .ir_type = PIPE_SHADER_IR_NIR,
    .prog = nir,
    .static_shared_mem = nir->info.shared_size,
  };
  void *cso = ctx->pipe->create_compute_state(ctx->pipe, &state);
  if (!cso) { ralloc_free(nir); errno = EINVAL; return NULL; }
  struct tm_kernel *kernel = calloc(1, sizeof(*kernel));
  if (!kernel) { ctx->pipe->delete_compute_state(ctx->pipe, cso); return NULL; }
  kernel->cso = cso;
  kernel->num_samplers = num_samplers;
  kernel->num_images = num_images;
  return kernel;
}

void tm_kernel_destroy(struct tm_context *ctx, struct tm_kernel *kernel) {
  if (!kernel) return;
  tm_synchronize(ctx);
  ctx->pipe->delete_compute_state(ctx->pipe, kernel->cso);
  free(kernel);
}

/* offsets[] points at the 64-bit buffer address slots in args. Gallium fills them. */
int tm_launch(struct tm_context *ctx, struct tm_kernel *kernel,
              struct tm_buffer **buffers, const uint32_t *offsets, const uint32_t *image_info, uint32_t num_buffers,
              void *args, uint32_t args_size,
              const uint32_t grid[3], const uint32_t block[3], uint32_t shared_mem) {
  struct pipe_resource *resources[num_buffers ? num_buffers : 1];
  struct pipe_resource *globals[num_buffers ? num_buffers : 1];
  uint32_t *handles[num_buffers ? num_buffers : 1];
  for (uint32_t i = 0; i < num_buffers; i++) {
    if (!buffers[i] || offsets[i] + sizeof(uint64_t) > args_size) return -EINVAL;
    resources[i] = globals[i] = buffers[i]->res;
    handles[i] = (uint32_t *)((uint8_t *)args + offsets[i]);
    *(uint64_t *)handles[i] += buffers[i]->offset;
  }
  struct pipe_sampler_view *sviews[kernel->num_samplers ? kernel->num_samplers : 1];
  struct pipe_image_view iviews[kernel->num_images > kernel->num_samplers ? kernel->num_images - kernel->num_samplers : 1];
  uint32_t image_idx = 0;
  for (uint32_t i = 0; i < num_buffers; i++) {
    uint32_t width = image_info[4*i], height = image_info[4*i+1], itemsize = image_info[4*i+2];
    if (!itemsize) continue;
    enum pipe_format format = itemsize == 2 ? PIPE_FORMAT_R16G16B16A16_FLOAT : PIPE_FORMAT_R32G32B32A32_FLOAT;
    struct pipe_tex2d_from_buf tex2d = {
      .offset = buffers[i]->offset / (4 * itemsize),
      .row_stride = width,
      .width = width,
      .height = height,
    };
    if (image_idx < kernel->num_samplers) {
      struct pipe_sampler_view tmpl;
      u_sampler_view_default_template(&tmpl, buffers[i]->res, format);
      tmpl.is_tex2d_from_buf = true;
      tmpl.u.tex2d_from_buf = tex2d;
      sviews[image_idx] = ctx->pipe->create_sampler_view(ctx->pipe, buffers[i]->res, &tmpl);
      if (!sviews[image_idx]) return -ENOMEM;
    } else {
      iviews[image_idx - kernel->num_samplers] = (struct pipe_image_view) {
        .resource = buffers[i]->res,
        .format = format,
        .access = PIPE_IMAGE_ACCESS_READ_WRITE | PIPE_IMAGE_ACCESS_TEX2D_FROM_BUFFER,
        .shader_access = PIPE_IMAGE_ACCESS_READ_WRITE,
        .u.tex2d_from_buf = tex2d,
      };
    }
    image_idx++;
  }

  struct pipe_constant_buffer cb = {
    .buffer = NULL,
    .buffer_offset = 0,
    .buffer_size = args_size,
    .user_buffer = args,
  };
  ctx->pipe->bind_compute_state(ctx->pipe, kernel->cso);
  if (kernel->num_samplers) {
    struct pipe_sampler_state ss = {
      .wrap_r = PIPE_TEX_WRAP_CLAMP_TO_BORDER,
      .wrap_s = PIPE_TEX_WRAP_CLAMP_TO_BORDER,
      .wrap_t = PIPE_TEX_WRAP_CLAMP_TO_BORDER,
      .min_img_filter = PIPE_TEX_FILTER_NEAREST,
      .mag_img_filter = PIPE_TEX_FILTER_NEAREST,
      .unnormalized_coords = true,
    };
    void *sstate = ctx->pipe->create_sampler_state(ctx->pipe, &ss);
    ctx->pipe->bind_sampler_states(ctx->pipe, MESA_SHADER_COMPUTE, 0, 1, &sstate);
    ctx->pipe->delete_sampler_state(ctx->pipe, sstate);
    ctx->pipe->set_sampler_views(ctx->pipe, MESA_SHADER_COMPUTE, 0, kernel->num_samplers, 0, sviews);
  }
  if (kernel->num_images > kernel->num_samplers)
    ctx->pipe->set_shader_images(ctx->pipe, MESA_SHADER_COMPUTE, 0, kernel->num_images - kernel->num_samplers, 0, iviews);
  const struct pipe_constant_buffer *cbp = args_size ? &cb : NULL;
  ctx->pipe->set_constant_buffer(ctx->pipe, MESA_SHADER_COMPUTE, 0, false, cbp);
  ctx->pipe->set_global_binding(ctx->pipe, 0, num_buffers, resources, handles);
  struct pipe_grid_info info = {
    .work_dim = 3,
    .block = {block[0], block[1], block[2]},
    .grid = {grid[0], grid[1], grid[2]},
    .variable_shared_mem = shared_mem,
    .globals = globals,
    .num_globals = num_buffers,
  };
  ctx->pipe->launch_grid(ctx->pipe, &info);
  ctx->pipe->set_global_binding(ctx->pipe, 0, num_buffers, NULL, NULL);
  for (uint32_t i = 0; i < kernel->num_samplers; i++) pipe_sampler_view_release(sviews[i]);
  return 0;
}
