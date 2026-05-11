# 1. Problem

small llm pretraining: llama 3.1 8b on c4.

## Requirements

Install tinygrad and mlperf-logging (uncomment mlperf from setup.py) from branch mlperf_training_v6.0.
```
git clone https://github.com/tinygrad/tinygrad.git
python3 -m pip install -e ".[mlperf]"
```

# 2. Directions

## Steps to download and verify data

### 1. Download raw data

follow mlperf steps to download the preprocessed c4 dataset.

## Running

### tinybox_8xMI350X

#### Steps to run benchmark
```
examples/mlperf/training_submission_v6.0/tinycorp/benchmarks/llama31_8b/implementations/tinybox_8xMI350X/run_and_time.sh
```
