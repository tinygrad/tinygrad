/*
 * Copyright 2017 Red Hat Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * Authors: Karol Herbst <kherbst@redhat.com>
 */

#define NVISA_G80_CHIPSET      0x50
#define NVISA_GF100_CHIPSET    0xc0
#define NVISA_GK104_CHIPSET    0xe0
#define NVISA_GK20A_CHIPSET    0xea
#define NVISA_GM107_CHIPSET    0x110
#define NVISA_GM200_CHIPSET    0x120
#define NVISA_GV100_CHIPSET    0x140

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"

/*
#include "util/u_debug.h"
#include "util/u_prim.h"
*/

/*
#include "nv50_ir.h"
#include "nv50_ir_lowering_helper.h"
#include "nv50_ir_target.h"
#include "nv50_ir_util.h"
#include "tgsi/tgsi_from_mesa.h"
*/

static nir_shader_compiler_options
nvir_nir_shader_compiler_options(int chipset, uint8_t shader_type)
{
   nir_shader_compiler_options op = {};
   op.lower_fdiv = (chipset >= NVISA_GV100_CHIPSET);
   op.lower_ffma16 = false;
   op.lower_ffma32 = false;
   op.lower_ffma64 = false;
   op.fuse_ffma16 = false; /* nir doesn't track mad vs fma */
   op.fuse_ffma32 = false; /* nir doesn't track mad vs fma */
   op.fuse_ffma64 = false; /* nir doesn't track mad vs fma */
   op.lower_flrp16 = (chipset >= NVISA_GV100_CHIPSET);
   op.lower_flrp32 = true;
   op.lower_flrp64 = true;
   op.lower_fpow = true;
   op.lower_fsat = false;
   op.lower_fsqrt = false; // TODO: only before gm200
   op.lower_sincos = false;
   op.lower_fmod = true;
   op.lower_bitfield_extract = (chipset >= NVISA_GV100_CHIPSET || chipset < NVISA_GF100_CHIPSET);
   op.lower_bitfield_insert = (chipset >= NVISA_GV100_CHIPSET || chipset < NVISA_GF100_CHIPSET);
   op.lower_bitfield_reverse = (chipset < NVISA_GF100_CHIPSET);
   op.lower_bit_count = (chipset < NVISA_GF100_CHIPSET);
   op.lower_ifind_msb = (chipset < NVISA_GF100_CHIPSET);
   op.lower_find_lsb = (chipset < NVISA_GF100_CHIPSET);
   op.lower_uadd_carry = true; // TODO
   op.lower_usub_borrow = true; // TODO
   op.lower_mul_high = false;
   op.lower_fneg = false;
   op.lower_ineg = false;
   op.lower_scmp = true; // TODO: not implemented yet
   op.lower_vector_cmp = false;
   op.lower_bitops = false;
   op.lower_isign = (chipset >= NVISA_GV100_CHIPSET);
   op.lower_fsign = (chipset >= NVISA_GV100_CHIPSET);
   op.lower_fdph = false;
   op.fdot_replicates = false; // TODO
   op.lower_ffloor = false; // TODO
   op.lower_ffract = true;
   op.lower_fceil = false; // TODO
   op.lower_ftrunc = false;
   op.lower_ldexp = true;
   op.lower_pack_half_2x16 = true;
   op.lower_pack_unorm_2x16 = true;
   op.lower_pack_snorm_2x16 = true;
   op.lower_pack_unorm_4x8 = true;
   op.lower_pack_snorm_4x8 = true;
   op.lower_unpack_half_2x16 = true;
   op.lower_unpack_unorm_2x16 = true;
   op.lower_unpack_snorm_2x16 = true;
   op.lower_unpack_unorm_4x8 = true;
   op.lower_unpack_snorm_4x8 = true;
   op.lower_pack_split = false;
   op.lower_extract_byte = (chipset < NVISA_GM107_CHIPSET);
   op.lower_extract_word = (chipset < NVISA_GM107_CHIPSET);
   op.lower_insert_byte = true;
   op.lower_insert_word = true;
   op.vertex_id_zero_based = false;
   op.lower_base_vertex = false;
   op.lower_helper_invocation = false;
   op.optimize_sample_mask_in = false;
   op.lower_cs_local_index_to_id = true;
   op.lower_cs_local_id_to_index = false;
   op.lower_device_index_to_zero = true;
   op.lower_wpos_pntc = false; // TODO
   op.lower_hadd = true; // TODO
   op.lower_uadd_sat = true; // TODO
   op.lower_usub_sat = true; // TODO
   op.lower_iadd_sat = true; // TODO
   op.lower_to_scalar = false;
   op.unify_interfaces = false;
   op.lower_mul_2x32_64 = true; // TODO
   op.has_rotate32 = (chipset >= NVISA_GV100_CHIPSET);
   op.has_imul24 = false;
   op.has_fmulz = (chipset > NVISA_G80_CHIPSET);
   op.intel_vec4 = false;
   op.lower_uniforms_to_ubo = true;
   op.force_indirect_unrolling = (nir_variable_mode) (
      ((shader_type == MESA_SHADER_FRAGMENT) ? nir_var_shader_out : 0) |
      /* HW doesn't support indirect addressing of fragment program inputs
       * on Volta.  The binary driver generates a function to handle every
       * possible indirection, and indirectly calls the function to handle
       * this instead.
       */
      ((chipset >= NVISA_GV100_CHIPSET && shader_type == MESA_SHADER_FRAGMENT) ? nir_var_shader_in : 0)
   );
   op.force_indirect_unrolling_sampler = (chipset < NVISA_GF100_CHIPSET);
   op.max_unroll_iterations = 32;
   op.lower_int64_options = (nir_lower_int64_options) (
      ((chipset >= NVISA_GV100_CHIPSET) ? nir_lower_imul64 : 0) |
      ((chipset >= NVISA_GV100_CHIPSET) ? nir_lower_isign64 : 0) |
      nir_lower_divmod64 |
      ((chipset >= NVISA_GV100_CHIPSET) ? nir_lower_imul_high64 : 0) |
      ((chipset >= NVISA_GV100_CHIPSET) ? nir_lower_bcsel64 : 0) |
      ((chipset >= NVISA_GV100_CHIPSET) ? nir_lower_icmp64 : 0) |
      ((chipset >= NVISA_GV100_CHIPSET) ? nir_lower_iabs64 : 0) |
      ((chipset >= NVISA_GV100_CHIPSET) ? nir_lower_ineg64 : 0) |
      ((chipset >= NVISA_GV100_CHIPSET) ? nir_lower_logic64 : 0) |
      ((chipset >= NVISA_GV100_CHIPSET) ? nir_lower_minmax64 : 0) |
      ((chipset >= NVISA_GV100_CHIPSET) ? nir_lower_shift64 : 0) |
      nir_lower_imul_2x32_64 |
      ((chipset >= NVISA_GM107_CHIPSET) ? nir_lower_extract64 : 0) |
      nir_lower_ufind_msb64 |
      ((chipset >= NVISA_GV100_CHIPSET) ? nir_lower_conv64 : 0)
   );
   op.lower_doubles_options = (nir_lower_doubles_options) (
      ((chipset >= NVISA_GV100_CHIPSET) ? nir_lower_drcp : 0) |
      ((chipset >= NVISA_GV100_CHIPSET) ? nir_lower_dsqrt : 0) |
      ((chipset >= NVISA_GV100_CHIPSET) ? nir_lower_drsq : 0) |
      ((chipset >= NVISA_GV100_CHIPSET) ? nir_lower_dfract : 0) |
      nir_lower_dmod |
      ((chipset >= NVISA_GV100_CHIPSET) ? nir_lower_dsub : 0) |
      ((chipset >= NVISA_GV100_CHIPSET) ? nir_lower_ddiv : 0)
   );
   op.discard_is_demote = true;
   op.has_ddx_intrinsics = true;
   op.scalarize_ddx = true;
   op.support_indirect_inputs = (uint8_t)BITFIELD_MASK(MESA_SHADER_GEOMETRY + 1);
   op.support_indirect_outputs = (uint8_t)BITFIELD_MASK(MESA_SHADER_GEOMETRY + 1);

   /* HW doesn't support indirect addressing of fragment program inputs
    * on Volta.  The binary driver generates a function to handle every
    * possible indirection, and indirectly calls the function to handle
    * this instead.
    */
   if (chipset < NVISA_GV100_CHIPSET)
      op.support_indirect_outputs |= BITFIELD_BIT(MESA_SHADER_FRAGMENT);

   return op;
}

int main(void) {
  fprintf(stderr, "size: %ld\n", sizeof(nir_shader_compiler_options ));
  nir_shader_compiler_options ops = nvir_nir_shader_compiler_options(NVISA_GV100_CHIPSET, MESA_SHADER_COMPUTE);
  return write(1, &ops, sizeof(nir_shader_compiler_options));
}

