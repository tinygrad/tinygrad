panfrost_symbols = [
  'panfrost_drm_screen_create',
  'pan_shader_get_compiler_options',
  'driParseOptionInfo',
  'driDestroyOptionCache',
  'driDestroyOptionInfo',
  'glsl_type_singleton_init_or_ref',
  'glsl_type_singleton_decref',
]

panfrost_link_args = []
foreach sym : panfrost_symbols
  panfrost_link_args += '-Wl,-u,' + sym
endforeach

shared_library('tinymesa',
  link_whole : [ _libnir, libgallium ],
  link_args : panfrost_link_args,
  dependencies : [ driver_panfrost, idep_xmlconfig ],
)
