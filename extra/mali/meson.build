shared_library('tinymali',
  'tinymali.c',
  include_directories : [inc_src, inc_gallium, inc_gallium_aux, inc_compiler, inc_panfrost, inc_include, include_directories('../util'), include_directories('../compiler/nir')],
  link_whole : [libgallium, _libnir],
  dependencies : [driver_panfrost, dep_libdrm, idep_xmlconfig],
  install : false,
)
