# Phase-2 LLM serving configurations

All configurations use redqwenflash with a recorded `DEPLOYED_COMMIT`, 262144 max context, and an OpenAI API. Start/stop/status wrappers are in `tools/launch/C1.sh` through `C9.sh` in the operations repo. Expected and measured results are in **[MATRIX.md](http://192.168.40.68:8090/)**; never treat unmeasured cells as zero.

Weights live below `/raid/weights` (C5 uses `/raid/models`). On ts1, GLM IQ4 uses the nested alternative `.../GLM-5.3-Flash-UD-IQ4_XS/UD-IQ4_XS/<shards>`; launchers accept either layout. Set `MODEL` to the first `.gguf` shard from the row, then run the environment shown.

## Qwen Flash

Common environment:

```sh
export PYTHONPATH=. DEV=AMD AM_MEC_PIPES=2 SPEC_DRAFTS=2 AM_IDLE_MS=1000
export QSA_KV_BITS=4 QSA_KV_ROT=3 QSA_KV_SPLIT=1 QSA_IDX_SPLIT=1 PREFILL_MB1_ROWS=1200
```

| ID | Model / weights (`MODEL`) | Box | Extra environment |
|---|---|---|---|
| C1 | `Qwen3.8-Flash-Next-UD-IQ3_XXS/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf` | tr8/tr9 | `AM_POWER_LIMIT=220 SEQ_ROWS=4798464 SEQ_SLOTS=64 SEQ_STASH_GB=55 SEQ_SIZES=1,4,8,12,16,20,24,28,32,40,48,64` |
| C2 | `Qwen3.8-Flash-Next-UD-IQ4_XS/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` | tr4 prod / tr8 | `AM_POWER_LIMIT=220 SEQ_ROWS=3262464 SEQ_SLOTS=64 SEQ_STASH_GB=55 SEQ_SIZES=1,4,8,12,16,20,24,28,32,40,48` |
| C3 | C1 weights | ts1 | `AM_POWER_LIMIT=210 SEQ_ROWS=5342208 SEQ_SLOTS=96 SEQ_STASH_GB=100 SEQ_SIZES=1,4,8,12,16,20,24,28,32,40,48,64` |
| C4 | C2 weights | ts1 | `AM_POWER_LIMIT=210 SEQ_ROWS=4614144 SEQ_SLOTS=96 SEQ_STASH_GB=100 SEQ_SIZES=1,4,8,12,16,20,24,28,32,40,48,64` |

Run from the deployed tree:

```sh
python3 -m tinygrad.llm --model "$MODEL" --shard 6 --max_context 262144 --serve 8000
```

## GLM Flash and GLM-5.3

Common environment:

```sh
export PYTHONPATH=. PYTHONUNBUFFERED=1 MLA_SEQ_SHARD=1 WARMUP_PARALLEL=1
```

| ID | Weights (`MODEL`) | Box | Extra environment | Command |
|---|---|---|---|---|
| C5 | `/raid/models/GLM-5.3-Flash-UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf` | tr9 | `DEV=AMD AM_POWER_LIMIT=220 BATCH_SLOTS=8` | `--shard 6` |
| C6 | `/raid/weights/GLM-5.3-Flash-UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf` | ts5 half / ts1 | `DEV=PCI:0-5+AMD VFIO=1 AM_POWER_LIMIT=210 BATCH_SLOTS=32` | `--shard 6` |
| C7 | `/raid/weights/GLM-5.3-Flash-UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf` | ts5 half / ts1 | `DEV=PCI:0-5+AMD VFIO=1 AM_POWER_LIMIT=210 BATCH_SLOTS=32` | `--shard 6` |
| C8 | `/raid/weights/GLM-5.3-UD-IQ3_XXS/GLM-5.3-UD-IQ3_XXS-00001-of-00007.gguf` | ts5 | `DEV=PCI+AMD VFIO=1 AM_POWER_LIMIT=184 SPEC_DRAFTS=3` | `--shard 14` |
| C9 | `/raid/weights/GLM-5.3-UD-IQ4_XS/GLM-5.3-UD-IQ4_XS-00001-of-00009.gguf` | ts5 | `DEV=PCI+AMD VFIO=1 AM_POWER_LIMIT=210 SPEC_DRAFTS=3` | `--shard 14` |

Use each row's environment and shard value with the equivalent command:

```sh
python3 -m tinygrad.llm --model "$MODEL" --shard <from-row> --max_context 262144 --serve 8000
```

For a second disjoint ts5 half use `DEV=PCI:6-11+AMD`, `--shard 6`, and `--serve 8001`; only the box owner may schedule it. Every R9700 remains at or below 210 W/GPU. The 184 W C8 default is conservative, not a required minimum. RDNA3 uses 220 W normally; 300 W is allowed only for an explicitly recorded peak measurement.
