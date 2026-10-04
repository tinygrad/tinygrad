# GPT-OSS 20B, MLPerf Training v6.1, single-node 8xMI350X

Run from the tinygrad repository root with all eight GPUs available and the
prepared C4 8B data at `/raid/datasets/c4-8b/`. This uses tinygrad's userspace AMD
driver and the current HCQ2/DP8/BS16, two-update-capture performance recipe.
Two updates per capture are **not** gradient accumulation: every minibatch updates
the weights and scheduler. The independent launcher retains the settings from
`benchmarks/gpt_oss/implementations/tinybox_8xMI350X/dev_run.sh`; parity is tested.
The submission directory and logged benchmark use the official name `gpt_oss_20b`.

## Install and run

Install the exact [6.1.0-rc2 release](https://github.com/mlcommons/logging/releases/tag/6.1.0-rc2),
commit `8a0cc0558ab498b3b003b106d7eae0de19801876`. The release's internal Python
package version is still `4.1.67`, so the requirement pins the Git commit.

```sh
python3 -m pip install -r examples/mlperf/training_submission_v6.1/tinycorp/benchmarks/gpt_oss_20b/implementations/tinybox_8xMI350X/requirements.txt
bash examples/mlperf/training_submission_v6.1/tinycorp/benchmarks/gpt_oss_20b/implementations/tinybox_8xMI350X/run_and_time.sh
```

`WANDB=1` enables tracking for both warmup and the measured run. Like Llama, the launcher sets
`SEED=$RANDOM` and `DATA_SEED=$SEED` once per invocation, shared by warmup and
training. Both values are logged; the batch launcher does not supply seeds.

Each invocation resets the GPU hive before warmup; a failed reset stops the run.

Like the Llama submission, the wrapper first clears its compilation cache and
runs a full 24-layer, ten-update fake-data initialization/BEAM phase, including
evaluation. Initialization remains open across the process boundary. The second
process initializes the full 24-layer model, optimizer, and train/evaluation
dataset indices before emitting `init_stop` followed immediately by `run_start`.
Dataset preparation reads metadata only; token `.bin` files are lazily mapped
on the first sample read. Token reads, any prefetch worker startup, and training
graph capture remain **inside** `run_start`/`run_stop`, matching the Llama timing
boundary. This does not move actual training or real-data evaluation into initialization.
`run_stop: success` is emitted only after a complete 1024-sequence evaluation
actually meets the loss target; nonconvergence or a loop exception logs `aborted`.
Evaluation `samples_count` is cumulative training **sequences**, not tokens,
microbatches, or captures, as required by the release's RCP reader.

The launcher follows Llama's structure: settings, logging, warmup, then training.
It uses the default compilation cache or an explicitly supplied `CACHEDB`;
the initialization phase clears that selected cache. Run without diagnostic
environment overrides such as `FAKEDATA`, `BENCHMARK`, or reduced `LAYERS`.

Both phases append to `result_gpt_oss_20b_<seed>.txt`, or the path supplied in
`MLPERF_LOGFILE`. Choose a fresh result path for each run; existing files are
appended to, and custom parent directories must already exist. Console output
is retained separately in a timestamped `.log`. Failures propagate through
`tee`. Install dependencies beforehand and run the checkers separately below.

Precision disclosure is FP8 for linear operations (MXFP8 is represented by the
`fp8` enum), BF16 for attention, and FP8 for communication in the production
ZeRO configuration: communication includes quantized weight all-gathers as well
as BF16 gradient transfers. Tensor/pipeline/context/expert parallelism are all 1;
the local microbatch is 2 on each of eight data-parallel GPUs.

## Check logs and package

Use the same pinned Python environment for all checkers:

```sh
python3 -m mlperf_logging.compliance_checker --usage training --ruleset 6.1.0 --werror /path/to/result_gpt_oss_20b_SEED.txt
python3 -m mlperf_logging.system_desc_checker examples/mlperf/training_submission_v6.1/tinycorp/systems/tinybox_8xMI350X.json training 6.1.0
python3 -m mlperf_logging.repo_checker examples/mlperf/training_submission_v6.1/tinycorp training 6.1.0
```

For the final package, put **ten real runs** under
`tinycorp/results/tinybox_8xMI350X/gpt_oss_20b/result_<index>.txt`. Keep console logs,
checker output, caches, and synthetic test fixtures out of that results directory.
Do not reuse or relabel historical runs without the required original events.
Review the system description against the final machine/software inventory;
schema validation does not verify its hardware claims or final source revision.

```sh
python3 -m mlperf_logging.package_checker examples/mlperf/training_submission_v6.1/tinycorp training 6.1.0 --werror
python3 -m mlperf_logging.rcp_checker examples/mlperf/training_submission_v6.1/tinycorp/results/tinybox_8xMI350X/gpt_oss_20b --rcp_version 6.1.0
python3 -m mlperf_logging.result_summarizer examples/mlperf/training_submission_v6.1/tinycorp training 6.1.0
```

Inspect the checker output: some tools in this upstream release print failure
without returning a nonzero shell status. Our package regression test asserts
their underlying boolean results, including seed and RCP checks with no bypass.
The repository checker's `run_checks()` also returns `False` even when it prints
`ALL CHECKS PASSED`; the regression test verifies its individual checks and report.

## CPU regression tests

```sh
DEV=CPU python3 -m pytest test/unit/test_gptoss_mlperf_logging.py test/unit/test_gptoss_launchers.py test/unit/test_binidx_dataset.py test/unit/test_gptoss_prefetch.py -x -q -n12
```

These exercise the actual training-loop logging with mocked model/data outputs,
phase isolation, setup-before-timing and token-read-after-timing ordering,
error/NaN/partial-evaluation handling, and official compliance,
system-description, repository and package (seed/RCP) checkers plus result summarization. Package fixtures use explicitly
synthetic convergence counts and temporary directories; they are never submission
results. Passing these tests does **not** establish numerical convergence or final
package acceptance. Ten measured runs and their real RCP check remain necessary.
