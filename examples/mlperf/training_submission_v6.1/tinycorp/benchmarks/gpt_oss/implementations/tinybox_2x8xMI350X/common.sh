# production 2x8 MI350X configuration shared by dev_beam.sh, dev_run.sh and run_and_time.sh
# two boxes: this box's gpus and nics opened directly ("local", AMD:0-7), the far box behind its serve.py (setup.sh up there, AMD:8-15)
export NODES=${NODES:-"tinyamd3 tinyamd4"}
export REMOTE=${REMOTE:-"local,$(echo $NODES | tr ' ' '
' | grep -vx "$(hostname)" | head -1):6667"}
export DEV=PCI+AMD REMOTE_TIMEOUT=600 ALLREDUCE_NODE_NDEVS=8 DP=16 BS=32 EVAL_BS=${EVAL_BS:-32} EVAL_SAMPLES=${EVAL_SAMPLES:-1024}
export AMD_AQL=1 PYTHONUNBUFFERED=1 DEBUG=${DEBUG:-0}
# expert weight all-gather straight from the owned shard buffers (no staging copy)
export GPTOSS_OWNED_EXPERT_GATHER=${GPTOSS_OWNED_EXPERT_GATHER:-1}
# collectives: node-leader small-message allreduce, duplex rdma with receives posted ahead, held copies
export ALLREDUCE_NODE_LEADER=${ALLREDUCE_NODE_LEADER:-1} RDMA_DUPLEX=${RDMA_DUPLEX:-1} RDMA_POST_AHEAD=${RDMA_POST_AHEAD:-1} COPY_PREHOLD=${COPY_PREHOLD:-1000}
export GPTOSS_DEFER_EXPERT_GATHER=${GPTOSS_DEFER_EXPERT_GATHER:-2} GPTOSS_DEFER_FC1_REDUCE=${GPTOSS_DEFER_FC1_REDUCE:-1}
# schedule: nic sends at their need, the next forward interleaved with the gather assemblies
export GPTOSS_SCHED_FIXER=${GPTOSS_SCHED_FIXER:-1} GPTOSS_NIC_NEED=${GPTOSS_NIC_NEED:-1}
export GPTOSS_FIRST_NEED=${GPTOSS_FIRST_NEED:-1} GPTOSS_FWD_INTERLEAVE=${GPTOSS_FWD_INTERLEAVE:-1}
# the graph's last update stages its expert shards and the next graph gathers them under its first forward (start region; the eval
# drain is a captured graph), the optimizer paced 60 calls ahead of its forward consumer, the lm head gather late: -19 ms/update (2x8 A/B)
export GPTOSS_START_REGION=${GPTOSS_START_REGION:-1} GPTOSS_JIT_OPT=${GPTOSS_JIT_OPT:-60} GPTOSS_LMHEAD_LATE=${GPTOSS_LMHEAD_LATE:-150}
# kernels: attention weights stored pre-padded, routing metadata in one hip kernel
export GPTOSS_PAD_ATTN_W=${GPTOSS_PAD_ATTN_W:-1} ROUTE_META_HIP=${ROUTE_META_HIP:-1}
# residual join from the saved attention output (keeps the fused residual+norm fwd with padded attn weights), custom-kernel outputs
# bound without copies, rmsnorm+quantize fwd at 16384 workgroups: -7 ms/update (1x8 ta3 hw A/B, losses within atomics noise)
export GPTOSS_RESID_FROM_SAVED=${GPTOSS_RESID_FROM_SAVED:-1} OWNED_KERNEL_OUTPUTS=${OWNED_KERNEL_OUTPUTS:-1} GPTOSS_RMSNORM_MX_FWD_WG=${GPTOSS_RMSNORM_MX_FWD_WG:-16384}
# 2 updates per captured graph, captured on the first call (no eager warmup update)
export GPTOSS_STEP_GROUP=${GPTOSS_STEP_GROUP:-2} GPTOSS_JIT_NO_WARMUP=${GPTOSS_JIT_NO_WARMUP:-1} JITBEAM=${JITBEAM:-3}
# peak LR 4.5e-4 with a 128-update warmup (fit on our + v6.0 eval curves: ~34% of runs at 18 evals, ~56% at 19, RCP fail risk ~4%), end LR = 0.1x per the rules, weight decay on every parameter like the reference,
# the required 12,288-sequence evaluation cadence
export LR=${LR:-0.00045} END_LR=${END_LR:-0.000045} WARMUP_STEPS=${WARMUP_STEPS:-128} GPTOSS_REF_WD=${GPTOSS_REF_WD:-1}
export EVAL_FREQ=${EVAL_FREQ:-12288} CKPT=${CKPT:-0} LOGMLPERF=${LOGMLPERF:-1}
# SAMPLES inherits the single-node launcher default: MAX_STEPS * GBS (38.4M sequences at BS=32).
# a fresh tinygrad cache unless the caller chose one (dev_beam.sh clears whatever cache it is given)
export CACHEDB=${CACHEDB:-$(mktemp -d)/cache.db}
# the captured training and eval graphs, saved by dev_beam.sh and loaded by dev_run.sh
export GPTOSS_JIT_PKL=${GPTOSS_JIT_PKL:-$HOME/gptoss_jit.pkl}
