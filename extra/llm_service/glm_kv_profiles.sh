#!/usr/bin/env bash
# C9's frontend GLM_KV alias only. The engine reads the following explicit controls.
apply_glm_kv_profile() {
  local profile=${GLM_KV:-half}
  case "$profile" in
    half) ;;
    q4)
      [[ ${SEQ_FP8:-1} == 1 && ${SEQ_FP8_IDX:-0} == 0 && ${SEQ_KV4:-1} == 1 ]] || {
        echo 'GLM_KV=q4 requires SEQ_FP8=1, SEQ_FP8_IDX=0 and SEQ_KV4=1 (defaults are allowed)' >&2
        return 2
      }
      export GLM_KV=q4 SEQ_FP8=1 SEQ_FP8_IDX=0 SEQ_KV4=1
      ;;
    fp8)
      [[ ${SEQ_KV4:-0} == 0 ]] || {
        echo 'GLM_KV=fp8 conflicts with SEQ_KV4=1; use q4 for four-bit latent rows' >&2
        return 2
      }
      # Owner fp8 profile: packed latent rows, with a half-precision index by default.
      export GLM_KV=fp8 SEQ_FP8="${SEQ_FP8:-1}" SEQ_FP8_IDX="${SEQ_FP8_IDX:-0}"
      ;;
    *) echo "GLM_KV must be half, fp8 or q4, got $profile" >&2; return 2 ;;
  esac
}
