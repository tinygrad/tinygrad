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

For the submission wrapper, override only `DATASET_PATH`, `MODEL_PATH`, `TOKENIZER_PATH`, `SEED`, or `LOGFILE` as needed for the host and dataset layout. Use `dev_run.sh` for experiments that intentionally change benchmark shape, runtime knobs, or checkpoint behavior. `LLAMA_LAYERS` is only for the fake-data init warmup path and is scrubbed before the real benchmark run.
