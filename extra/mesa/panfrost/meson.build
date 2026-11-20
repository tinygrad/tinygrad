panfrost_symbols = [
  'panfrost_drm_screen_create',
  'panfrost_resource_address',
  'panfrost_nir_has_float_compare',
  'panfrost_compute_shader_desc',
  'panfrost_compute_shader_info',
  'panfrost_device_vm_id',
  'panfrost_compute_bind',
  'panfrost_compute_set_constants',
  'panfrost_compute_launch',
  'panfrost_compute_create_sampler_view',
  'panfrost_compute_set_samplers',
  'panfrost_compute_release_sampler_view',
  'panfrost_compute_set_images',
  'panfrost_compute_flush',
  'panfrost_buffer_write',
  'panfrost_buffer_read',
  'panfrost_buffer_copy',
  'panfrost_compute_wait',
  'panfrost_compute_signal',
  'panfrost_compute_timestamp',
  'pan_shader_get_compiler_options',
  'driParseOptionInfo',
  'driDestroyOptionCache',
  'driDestroyOptionInfo',
  'glsl_type_singleton_init_or_ref',
  'glsl_type_singleton_decref',
  'nir_to_spirv',
  'spirv_shader_delete',
]

panfrost_link_args = []
foreach sym : panfrost_symbols
  panfrost_link_args += '-Wl,-u,' + sym
endforeach

zink_compiler = static_library('zink_compiler',
  [ files('../gallium/drivers/zink/nir_to_spirv/nir_to_spirv.c',
          '../gallium/drivers/zink/nir_to_spirv/spirv_builder.c'), zink_device_info[0], zink_instance[0], spirv_info_h ],
  include_directories : [inc_include, inc_src, inc_mapi, inc_mesa, inc_gallium, inc_gallium_aux, inc_gallium_drivers, inc_vulkan_util, inc_zink_vk, include_directories('../gallium/drivers/zink')],
  dependencies : [idep_nir_headers, idep_mesautil, idep_vulkan_util_headers, idep_vulkan_wsi_defines, idep_vulkan_util, dep_libdrm, vulkan_wsi_deps],
)

shared_library('tinymesa',
  link_whole : [ _libnir, libgallium, zink_compiler ],
  link_args : panfrost_link_args,
  dependencies : [ driver_panfrost, idep_xmlconfig ],
)
