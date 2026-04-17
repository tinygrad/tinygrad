# 1. Problem

This benchmark uses Llama2 70B LoRA for MLPerf Training v6.0 language fine-tuning.

## Requirements

From the repository root, install tinygrad, MLPerf logging, and the optional parquet reader used by the GovReport dataset path.
```
python3 -m pip install -e .
git clone https://github.com/mlcommons/logging.git mlperf-logging
python3 -m pip install -e mlperf-logging
python3 -m pip install pyarrow sentencepiece tqdm
```

# 2. Directions

## Dataset and weights

The wrapper expects the SCROLLS GovReport 8k dataset at:
```
DATASET_PATH=/raid/datasets/scrolls_gov_report_8k
```

The directory may contain the MLPerf parquet files:
```
train-00000-of-00001.parquet
validation-00000-of-00001.parquet
```

It also accepts `train.jsonl` / `validation.jsonl` for local development.

The base model path must point at converted Llama2 70B weights loadable by `FlatTransformer.load_from_pretrained(...)`. By default:
```
MODEL_PATH=/raid/weights/llama2-70b-fused-qkv-mlperf
```

## Benchmark flow

`run_and_time.sh` performs two phases:
1. `INITMLPERF=1` on `FAKEDATA=1` with `LLAMA_LAYERS=2` and `MODEL_PATH=""` to warm up beam search and emit init logging without loading the full 70B checkpoint.
2. `RUNMLPERF=1` on the fixed submission shape with:
```
MP=8
DP=1
BS=1
EVAL_BS=1
GRADIENT_ACC_STEPS=1
SEQLEN=8192
MAX_STEPS=1024
EVAL_FREQ=48
EVAL_TARGET=0.925
LR=4e-4
END_LR=0.0
WARMUP_STEPS=0
LLAMA_LORA_RANK=16
LLAMA_LORA_ALPHA=32
LLAMA_LORA_DROPOUT=0.1
```
`run_and_time.sh` also forces `LOGMLPERF=1`, `SUBMISSION_PLATFORM=tinybox_8xMI350X`, `DATA_SEED=SEED`, and ignores ambient checkpoint/debug split modifiers such as `TRAIN_ON_VAL`, `SMALL`, `LOAD_CKPT`, `RESUME_CKPT`, and `LLAMA_LAYERS`.
It also pins the runtime/kernel knobs carried in `dev_run.sh`, including `DEBUG=0`, `HK_FLASH_ATTENTION=1`, `ASM_GEMM=1`, `OFFLOAD_OPTIM=1`, `JITBEAM=3`, and the `BEAM_*` limits, so the submission wrapper does not inherit them from the caller shell.

## Running

### tinybox_8xMI350X

#### Steps to run benchmark
```
examples/mlperf/training_submission_v6.0/tinycorp/benchmarks/llama2_70b_lora/implementations/tinybox_8xMI350X/run_and_time.sh
```

#### Direct development run
```
DATASET_PATH=/raid/datasets/scrolls_gov_report_8k \
MODEL_PATH=/raid/weights/llama2-70b-fused-qkv-mlperf \
examples/mlperf/training_submission_v6.0/tinycorp/benchmarks/llama2_70b_lora/implementations/tinybox_8xMI350X/dev_run.sh
```

For the submission wrapper, override only `DATASET_PATH`, `MODEL_PATH`, `TOKENIZER_PATH`, `SEED`, or `LOGFILE` as needed for the host and dataset layout. During `RUNMLPERF=1`, `DATASET_PATH` must be a dataset directory exposing both train and validation splits. Use `dev_run.sh` for experiments that intentionally change benchmark shape, runtime knobs, or checkpoint behavior. `LLAMA_LAYERS` is only for the fake-data init warmup path and is scrubbed before the real benchmark run. The wrapper also ignores ambient `TRAIN`, `CKPT`, `FP8`, `LOAD_CKPT`, `RESUME_CKPT`, `TRAIN_ON_VAL`, `SMALL`, `SAMPLES`, and `EVAL_SAMPLES` overrides. `LOGFILE` is written relative to the caller shell, while adapter checkpoints land under repo-root `./ckpts/`. MLPerf events are written separately to `result_llama2_70b_lora_<SEED>.log` in the repo root.

## Preflight gates

Do not hand the run to GPUs until all of these are true:
- `python3 -m unittest test.unit.test_mlperf_llama_submission_wrapper test.unit.test_llama_lora_train_wiring test.unit.test_mlperf_llama2_lora_eval test.unit.test_mlperf_lr_schedulers examples.mlperf.models.test_flat_llama`
  passes from the repo root.
- `DATASET_PATH` exists and exposes both train and validation splits (`train-*.parquet` / `validation-*.parquet` or matching json/jsonl files).
- `MODEL_PATH` points at converted fused-QKV Llama2 70B weights loadable by `FlatTransformer.load_from_pretrained(...)`.
- A tokenizer is discoverable from `MODEL_PATH/tokenizer.model` or `TOKENIZER_PATH` is set explicitly, and those files exist on disk.
- The target box exposes 8 AMD devices and has enough local disk for logs plus adapter checkpoints.

## GPU handover acceptance

I would only accept the implementation as successful after all of the following happen on the target GPUs:
1. `run_and_time.sh` completes the fake-data `INITMLPERF=1` phase and `result_llama2_70b_lora_<SEED>.log` contains `INIT_START` and `INIT_STOP`.
2. The real `RUNMLPERF=1` phase starts with the submission shape (`MP=8`, `DP=1`, `BS=1`, `EVAL_BS=1`, `SEQLEN=8192`, `MAX_STEPS=1024`) without manual script edits.
3. Base weights load from `MODEL_PATH`, the first training loop produces finite loss values, and at least one validation pass completes with a finite eval loss.
4. The MLPerf result log contains `RUN_START`, `EVAL_START/STOP`, and a terminal `RUN_STOP` status.

## Engineering hold points

These are separate from the default `run_and_time.sh` invocation, but I would still require them before calling the implementation done:
1. `dev_run.sh CKPT=1 ...` writes adapter-only checkpoints as both `llama2_70b_lora_<step>.safe` and `llama2_70b_lora_<step>_state.safe`.
2. `dev_run.sh CKPT=1 RESUME_CKPT=ckpts/llama2_70b_lora_<step>_state.safe ...` restarts cleanly, preserves step/sample accounting, and continues from the same next training batch.
3. The chosen `LOGFILE` captures stdout/stderr for both wrapper phases, while `result_llama2_70b_lora_<SEED>.log` remains the source of truth for MLPerf event verification.

Anything less than that is still a bring-up run, not proof that the implementation is done.
