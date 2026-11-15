# Running the three production inference services

Supported tree: `redqwenflash`, beginning at `5ce1f87eb2857a8c8d8cb3602dc7418f0fec4a79`; production deployments must record the exact commit in `DEPLOYED_COMMIT`. There are three products:

| Service | Model | Hardware | Context | Devices |
|---|---|---|---:|---|
| Qwen C2 | Qwen3.8 Flash Next UD-IQ4_XS **V3.1** + separate MTP + picture tower | 6x AMD 7900 XTX (RDNA3) | 262144 | `DEV=AMD`, 6 shards |
| GLM Flash C5 | GLM-5.3 Flash UD-IQ3_XXS, D34 quality configuration | 6x AMD 7900 XTX (RDNA3) | 18 x 262144; 524288 server cap | `DEV=AMD`, 6 shards |
| GLM C9 | GLM-5.3 UD-IQ4_XS | 14x AMD R9700 (RDNA4), ts5 | 262144 validated; 1048576 Phase-4 target pending | `DEV=PCI+AMD`, `VFIO=1`, 14 shards |

Only a machine's owner starts/stops service and only one GPU workload may use a box. Do not load the `amdgpu` kernel driver and do not run `modprobe`/DKMS: tinygrad's AM support drives the hardware. GPU jobs on shared infrastructure use its queue/ownership protocol. The three production launchers are `extra/llm_service/qwen_6x7900xtx.sh`, `glm_flash_6x7900xtx.sh` and `glm_14xr9700.sh` in this repository. VM/tunnel/proxy infrastructure is separate.

## Preconditions

1. Use the supported Python installed by the operations image, an immutable checkout, and `python3 -m pip install pillow>=10.0.1` for picture requests. Keep weights read-only and verify every hash below before a start.
2. Record a writable tinygrad cache (`XDG_CACHE_HOME`), persistent state/log directory, and ownership of the box. Never expose `SESSION_END_TOKEN` or API keys in argv, commits or logs. The HTTP service is loopback HTTP; TLS and external access belong to the front proxy.
3. Check all expected cards and free device/host memory. R9700 limit: **PENDING: power verification is in progress; the former launcher default 210 W is NOT safe. `GLM53_POWER_LIMIT` is required, with no numeric default until verification establishes one.** Qwen ran at **250 W/card** and GLM Flash D34 at **225 W/card**; use **225 W for Qwen on a hot host**, keep every hotspot **under 100 C**, and log it. A dropped/hung card is an incident, not a reset loop.

## Immutable weights and hashes

### Qwen C2 production set — V3.1, four shards

`MODEL` names shard 00001; the launcher verifies all four files against the manifest. Root: `/weights/Qwen3.8-Flash-Next-UD-IQ4_XS-V31`.

```text
UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf
UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00004.gguf
UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00004.gguf
UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00004-of-00004.gguf
MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
mmproj/mmproj-F16.gguf
```

```text
93ff23a08e5cd7501f24fa16f5b415d5807d0175e04edd3ed6a7a41b601a4eef  UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf
13d5dc8b93d039e4cfade5a9af7037f10c97f7ec4e53f75bd2e9ae34aae54aeb  UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00004.gguf
f6b9b5934908a2043732269dd740155d0f921d0763a1dcdd3452246311651f87  UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00004.gguf
5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36  UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00004-of-00004.gguf
5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6  MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
1f7b7f0b984cf065c604360c29c8098362ed61b290db0ff12c6f360bb1a8a980  mmproj/mmproj-F16.gguf
```

The Qwen engine owner verified these hashes directly from the production V3.1 weight files. The non-V31 three-shard set with otherwise similar names is **superseded** and is deliberately absent from this page's start/manifest commands. Never substitute it or a different text checkpoint.

### GLM Flash C5 production set — IQ3_XXS, four shards

Root: `/weights/GLM-5.3-Flash-UD-IQ3_XXS/`. `MODEL` names shard 00001; the launcher verifies all four files against `extra/llm_service/glm-flash-iq3xxs.sha256`.

```text
632f4fabe03ea73d20c299c8ef00c4a16e47dd9ab3f1750a8585b92eb70dfa74  GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf
a4dba331fe92ffbd71317839cc1452c57f0b5b3f971a9e96ee5107f9a615cdac  GLM-5.3-Flash-UD-IQ3_XXS-00002-of-00004.gguf
5b728b5b263c662201ef4a4a2b12fb509fe230d012418d0ed37c220a8a5aa230  GLM-5.3-Flash-UD-IQ3_XXS-00003-of-00004.gguf
4b6d16e0d1381e1b4b7eeb6c9799671bf98153a995be12e75e3da93cd97981d2  GLM-5.3-Flash-UD-IQ3_XXS-00004-of-00004.gguf
```

These four SHA-256 values were computed from deployment files and match the independent Phase-2 reference. The recorded g1 production start selects this same four-shard model path; an argv path is not itself a fresh file hash, so verify the manifest before every start. Do not substitute the similarly named IQ4_XS five-shard or IQ1_M three-shard checkpoints.

### GLM C9 production set

Directory: `/raid/weights/GLM-5.3-UD-IQ4_XS/` on ts5. `MODEL=/raid/weights/GLM-5.3-UD-IQ4_XS/GLM-5.3-UD-IQ4_XS-00001-of-00009.gguf`; all nine shards must be present.

All nine hashes below were computed directly from the C9 model files on ts5 and verified by the GLM engine owner. This is local-owner verification, not comparison against a publisher checksum; stop immediately on a mismatch.

```text
a788356c69085aaa8f082b9cdedc195e957675a3984d7aac48a2318fea626b8b  GLM-5.3-UD-IQ4_XS-00001-of-00009.gguf
b407bfbcfad7c55637dc86c00823680fc1b38a21ebab4a90590146ee0921380b  GLM-5.3-UD-IQ4_XS-00002-of-00009.gguf
11b725386fe58e2cabf3744ecd72d0a1ea592a53c24e66c6b4cceb8d78976115  GLM-5.3-UD-IQ4_XS-00003-of-00009.gguf
ed08d196ef1e946c1752ce11a2cffa9cd08461a2e42e3f4c5b44cf98d20d0af7  GLM-5.3-UD-IQ4_XS-00004-of-00009.gguf
02eec3692331e46229e03d8703aba7ced83e29a6ba73f945f6f2e77b5731eed8  GLM-5.3-UD-IQ4_XS-00005-of-00009.gguf
22dffc0f6d50afe5f4487a9319f87829d8d17685ca5060120783140ed9f72538  GLM-5.3-UD-IQ4_XS-00006-of-00009.gguf
b4db9060e67578b0e0dded328a1efeee89a22392478f46d7694e9566e4075ff0  GLM-5.3-UD-IQ4_XS-00007-of-00009.gguf
2fa8907c85f5fc58ac0dc4cad79e7a285ab8f4826c0fcd4340baaeb508cce81d  GLM-5.3-UD-IQ4_XS-00008-of-00009.gguf
852372068198079ec14ac7c6cf811945d78581061616045954f5d5c1fd882b33  GLM-5.3-UD-IQ4_XS-00009-of-00009.gguf
```

For any service place the relevant lines in a manifest beside the files and run `sha256sum -c <manifest>`. Keep the manifest and the successful output with the deployment record.

## Start, readiness and stop

### Qwen C2, on its six-card serving VM

Work from the immutable tinygrad checkout. This is the full environment from the running V3.1 server (with `AM_VM_RESET=1` only on its first controlled start) and complete engine invocation:

```bash
setsid env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. PYTHONUNBUFFERED=1 HOME="$HOME" \
  DEV=AMD AM_MEC_PIPES=2 AM_IDLE_MS=1000 AM_POWER_LIMIT=250 \
  SPEC_DRAFTS=2 QSA_KV_BITS=8 QSA_KV_ROT=3 QSA_KV_SPLIT=1 QSA_IDX_SPLIT=1 QSA_KV_BITS4_LAYERS=39,43,47,48 \
  SEQ_ROWS=4310016 SEQ_SLOTS=64 SEQ_SIZES=1,8,16,24,30 SEQ_MAX_ACTIVE=30 SEQ_STASH_GB=46 \
  SEQ_CKPT_HOST=1 SEQ_CKPT2=1 SEQ_BASE_ALIGN=24 \
  PREFILL_CHUNK=2304 PREFILL_SIZES=1152 PREFILL_PARTS=2 PREFILL_MB1_ROWS=1200 PREFILL_PK_CTX=131072 \
  MICRO_BATCHES=1 PARALLEL=40 RING_SEL=0 FLAG_AR_POOLN=65536 RING_MAXW_DEC=262144 LOAD_APART=1 TOP_K=20 \
  CLOCK_FAST=2000 CLOCK_MANY=1900 CLOCK_PREFILL=2000 \
  VISION_MMPROJ=/weights/Qwen3.8-Flash-Next-UD-IQ4_XS-V31/mmproj/mmproj-F16.gguf \
  VISION_KERNELS=1 VISION_SLICE_MS=200 VISION_MAX_IMAGES=1024 VISION_CACHE_MB=1024 VISION_URL_CACHE_MB=512 \
  XDG_CACHE_HOME=/home/runner/vm-cache \
  python3 -m tinygrad.llm \
  --model /weights/Qwen3.8-Flash-Next-UD-IQ4_XS-V31/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf \
  --shard 6 --max_context 262144 --serve 8000 > qwen-c2.log 2>&1 & echo $! > qwen-c2.pid
```

Use 225 W on a hot host. Omit `AM_VM_RESET` for every ordinary start; on the first verified-clean driver start only, prepend `AM_VM_RESET=1`. Set `SESSION_END_TOKEN` separately in the process environment only if the admin endpoint is needed. `GET /v1/models` must identify Qwen and `GET /status` must report slots=64/max_active=30 with eight-bit-plus-late-four-bit cache capacity. Stop: `kill -TERM "$(cat qwen-c2.pid)"`; await process exit and release of all six cards. The equivalent supported wrapper is `extra/llm_service/qwen_6x7900xtx.sh start`, whose `status`/`stop` actions use its state dir.

### GLM Flash C5, on its six-card serving VM

D34 serves 18 concurrent users with 262144 context each (a 524288 server cap), Q4 KV, 18 warmed slots/active requests, 3,916,800 pooled rows and zero stash. The canonical complete environment is printed by the launcher without touching a GPU:

```bash
bash extra/llm_service/glm_flash_6x7900xtx.sh --print-command
bash extra/llm_service/glm_flash_6x7900xtx.sh start
```

The fixed engine invocation is `python3 -m tinygrad.llm --model /weights/GLM-5.3-Flash-UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf --shard 6 --max_context 524288 --serve 8000`. D34 ran at 225 W/card. Leave `SPEC_DRAFTS` and `SPEC_DRAFTS_BY_SIZE` unset: this model inherits two drafts. Use `AM_FIRST_START=1` only for the first controlled start, not ordinary restarts. `status` and `stop` use the launcher's state dir. This tree's C5 endpoint is **text-only**; do not add Qwen's picture tower settings.

With `SESSION_END_TOKEN` unset (the default), explicit backend session-end calls return the expected **403 `backend_session_end_disabled`**, not generation failures. An integrated proxy releases its own session bookkeeping, while backend conversations remain in bounded 18-slot residency until reuse, NONDROP eviction or restart; `ended_dropped=0` only counts processed explicit ends and is **not** proof of backend memory release. Actual coordinated cleanup requires a matching out-of-band token on proxy and backends plus active/completed/duplicate-end tests (including resident/stash drops, `already_ended` and later admissions); never expose the secret in `--print-command`, and enabling the endpoint alone does not move model-seat release before verifier execution.

### GLM C9, on ts5

The validated start below is 262144 context per user. Phase 4 requires 1048576 context per user, but that capacity profile, its power limit and its quality/stability gates are still owner work; do not claim this command is a validated 1M deployment. Do not start C9 with the former 210 W default. Set the power value only after the completed verification establishes it, then run the complete environment:

```bash
export GLM53_POWER_LIMIT=<GLM_OWNER_VERIFIED_WATTS>
sudo -n env HOME="$HOME" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. PYTHONUNBUFFERED=1 \
  DEV=PCI+AMD VFIO=1 AM_POWER_LIMIT="$GLM53_POWER_LIMIT" \
  MLA_SEQ_SHARD=1 WARMUP_PARALLEL=1 IQ2S_K256=1 \
  HOST_STASH=0 HOST_MIN=4096 SLOT_HOST=1 SLOT_CKPT=1 \
  SPEC_DRAFTS=2 SPEC_DRAFTS_BY_SIZE=1:2,2:2,4:3,8:3,16:3 SEQ_WARMUP_ONLY= \
  GLM_SEQ=1 GLM_ATTN_GRP=3 SEQ_GATHER_UNION=65536 P2P_AR_TIMEOUT_MS=20000 \
  SEQ_ROWS=268800 SEQ_SLOTS=32 SEQ_SIZES=1,4,8,16 SEQ_MAX_ACTIVE=32 \
  SEQ_STASH_GB=0 SEQ_STASH_ENTRIES=128 LOAD_DEFER=1 \
  PREFILL_CHUNK=4032 PREFILL_SIZES=1344,2688 PREFILL_BIG_ALONE=2 \
  MODEL_ALIASES=zai-org/GLM-5.3 MODEL_OWNED_BY=tinygrad XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache/tinygrad}" \
  python3 -m tinygrad.llm \
  --model /raid/weights/GLM-5.3-UD-IQ4_XS/GLM-5.3-UD-IQ4_XS-00001-of-00009.gguf \
  --shard 14 --max_context 262144 --serve 8000
```

C9 therefore uses 14 shards, exact half cache rows and the `zai-org/GLM-5.3` alias. Require all 14 cards, expected effective env, `GET /v1/models`, healthy `GET /status`, then a small text request. The supported wrapper is `GLM53_POWER_LIMIT=$GLM53_POWER_LIMIT extra/llm_service/glm_14xr9700.sh start`; its `stop` action SIGTERMs the recorded process group and confirms exit. Rollback uses the previous immutable tree and matching validated profile.

## API and production environment contract (77 names)

API sampling: a request without a temperature uses the model's GGUF `general.sampling.temp` and `top_p` (Qwen 3.8 Flash: 1.0 / 0.95); absent sampling metadata keeps the previous greedy behavior, explicit request values are unchanged, and startup logs report the defaults.

There are 58 settings from the C2/C9 cleanup survival registry plus 19 exposed by the C5/D34 production contract. Count: 39 launcher-contract settings, 26 operator knobs, 12 short diagnostics (77 total, under the 80-name target).

Any setting not listed here is not supported: its production value is compiled into the tree. Values below are launch defaults/current validated profiles, not permission to run arbitrary combinations. Capacity, sharding and cache-format changes require their own real-memory and text gates. Empty means disabled/unset.

### Shared C2/C5/C9 hardware/model launcher contract (22)

| Name | C2 | C5 | C9 | Operator changes |
|---|---|---|---|---|
| `PREFILL_BIG_ALONE` | default | model default | 2 | Isolation policy for large prefill chunks. |
| `PREFILL_CHUNK` | 2304 | 768 | 4032 | Maximum prefill tokens per chunk. |
| `PREFILL_MB1_ROWS` | 1200 | 1200 | model-fitting profile | Rows retained for single-microbatch prefill. |
| `PREFILL_PARTS` | 2 | 4 | 4 | Graph parts in one prefill. |
| `PREFILL_PK_CTX` | 131072 | n/a | n/a | Context where large-prompt prefill kernels change. |
| `PREFILL_SIZES` | 1152 | 192,768 | 1344,2688 | Additional prefill JIT shapes. |
| `QSA_KV_BITS4_LAYERS` | 39,43,47,48 (example) | n/a | n/a | Attention layers stored in four-bit rows. |
| `QWEN_MTP` | auto-discovered | n/a | n/a | Optional separate speculative-head GGUF path. |
| `SEQ_BASE_ALIGN` | 24 | n/a | n/a | Packed-cache region base alignment. |
| `SEQ_CKPT_HOST` | 1 | model default | model profile | Host storage of recurrent checkpoints. |
| `SEQ_GATHER_UNION` | default | default | 65536 | Position enabling sparse union gathers. |
| `SEQ_MAX_ACTIVE` | 30 | 18 | 32 | Concurrent admission cap. |
| `SEQ_ROWS` | 4310016 (example) | 3916800 | 268800 | Reserved cache rows for this capacity profile. |
| `SEQ_SIZES` | 1,8,16,24,30 | 1,2,4,8,12,13,18 | 1,4,8,16 | Warmed decode batch sizes. |
| `SEQ_SLOTS` | 64 | 18 | 32 | Warmed resident conversation slots. |
| `SEQ_STASH_ENTRIES` | derived from stash profile; unset in the running env | 128 | 128 | Host-stash entry cap. |
| `SEQ_STASH_GB` | 46 | 0 | 0 | Host-stash bytes per device. |
| `SPEC_DRAFTS` | 2 | unset (model default 2) | 2 | Speculative tokens except per-size override. |
| `SPEC_DRAFTS_BY_SIZE` | empty | empty | 1:2,2:2,4:3,8:3,16:3 | Draft count by active batch size. |
| `VFIO` | auto/unset | auto/unset | 1 | Device pass-through mode. |
| `AM_VM_RESET` | controlled first start only | controlled first start only | controlled first start only | Permission to reset AM hardware once; ordinary starts forbid it. |
| `VISION_MMPROJ` | picture servers set path | unset; text-only | n/a | Picture-projection GGUF location. |

### Additional C5/D34 hardware/model contracts (17)

| Name | D34 value | Operator changes |
|---|---|---|
| `MLA_SEQ_SHARD` | 1 | Shard MLA sequence work over the six devices. |
| `WARMUP_PARALLEL` | 1 | Parallelism for start-time JIT compilation. |
| `IQ2S_K256` | 1 | Model-specific IQ2 kernel packing selection. |
| `SLOT_HOST` | 1 | Host placement policy for sequence slots. |
| `SLOT_CKPT` | 1 | Checkpoint placement policy for sequence slots. |
| `BATCH_SLOTS` | 8 | Scheduler batch-slot capacity on this RDNA3 profile. |
| `LOAD_DEFER` | 1 | Deferred weight-load schedule used by these GLM profiles. |
| `FLAG_AR_MAXN` | 32768 | Collective mailbox allocation for this model/device profile. |
| `AMD_SCRATCH_MIN` | 576 | Minimum AM scratch reservation on this profile. |
| `AMD_SCRATCH_MAX` | 64 | Maximum AM scratch reservation on this profile. |
| `SEQ_COPY_SCRATCH` | 1 | Scratch capacity for host/device sequence copies. |
| `SEQ_IDX_SHARD` | 1 | Shard the D34 attention index over devices. |
| `SEQ_KV4` | 1 | Four-bit latent-KV cache format selected by D34. |
| `PREFILL_SLICE` | 1536 | Rows allowed in one scheduled prefill slice. |
| `SEQ_STARVE_ALT` | 1 | Alternative anti-starvation scheduling policy. |
| `RING_MAXW` | 32768 | Ring collective window for this six-card profile. |
| `RING_RS_LIKE_COPIES` | 0 | Copy routing around reduce-scatter-like collectives. |

### Operator knobs documented for operations (26)

| Name | Default/profile | Purpose |
|---|---|---|
| `API_KEY` | empty (open loopback) | Comma-separated Bearer keys; prefer file for real secrets. |
| `API_KEY_FILE` | unset | File containing one Bearer key per line. |
| `ENC_CACHE` | 256 | Maximum cached chat-template encodings. |
| `HOST_MIN` | 4096 | Minimum rows retained for host-cache policy. |
| `HOST_STASH` | 0 | Host-backed stash policy/capacity for GLM. |
| `KEEPALIVE` | 15 s | SSE keepalive interval. |
| `MODEL_ALIASES` | C9 `zai-org/GLM-5.3` | Additional `/v1/models` ids; empty disables. |
| `MODEL_OWNED_BY` | tinygrad | Owner metadata in `/v1/models`. |
| `P2P_AR_TIMEOUT_MS` | C9 20000 | Abort stuck peer allreduce before host signal timeout. |
| `SEQ_CKPT2` | empty/off; validated C2 arms use 1 | Keep two prompt-end checkpoints a sequence. |
| `SEQ_PROTECT` | 60 s | Eviction protection for active conversations. |
| `SEQ_RESERVE` | 4096 rows | Cache rows excluded from admission. |
| `SEQ_STASH_CTX` | 32768 | Minimum context eligible for stash retention. |
| `SEQ_STICKY` | 180 s | Idle-conversation residency grace period. |
| `SESSION_END_TOKEN` | required for admin endpoint | Bearer credential for session termination. |
| `VISION_CACHE_MB` | dedicated picture profile 1024 | Maximum host RAM for cached picture rows/profile features: this many MB stays resident when full, in addition to decoded pictures and engine state. Size a shared service from measured RSS, not this dedicated-box profile. |
| `VISION_URL_CACHE_MB` | dedicated picture profile 512 | Maximum host RAM for decoded pictures keyed by data-URL hash: up to this many MB avoids repeated decoding but remains resident beside the row cache and engine state. Reduce it for a shared service after measuring its working set. |
| `VISION_DECODERS` | 2 | Concurrent picture decode workers. |
| `VISION_DECODE_WAIT` | 30 s | Picture decode deadline. |
| `VISION_MAX_BODY_MB` | 32 | HTTP request body cap. |
| `VISION_MAX_BYTES` | 20000000 | Encoded bytes allowed for one picture. |
| `VISION_MAX_IMAGES` | dedicated picture profile 1024; many-customer default 8, typical explicit limit 8–32 | Maximum pictures per request. Each can consume up to the byte/pixel/token caps plus decode, tower and queue work; the dedicated 1024 limit admits far more concurrent work and cache pressure. |
| `VISION_MAX_PIXELS` | 40000000 | Decoded pixel cap per picture. |
| `VISION_MAX_TOKENS` | 1024 | Picture token-row/tower memory cap. |
| `VISION_SLICE_MS` | 300 default; validated service overrides exist | Target picture work per scheduler slice. |
| `WARMUP_JITS` | C5/D34 `all` | Named start-time JIT subset; `all` warms every supported production graph. |

### Short diagnostics list (12)

These are off unless diagnosing or validating. Never enable fault injection in customer service. `DEBUG_LOAD`, `MEM_TRACE`, `AMD_SCRATCH_LOG`, `KPROF` (with `KPROF_TOP=30`), `LIN_NAMES`, `TELEMETRY_LOG=<jsonl-path>`, `SERVE_TRACE=<trace-path>`, `DISABLE_AMD_KERNELS=1` for generic-kernel reference runs, `RING_RDNA4=1` only on an owned RDNA4 validation job, `SEQ_WARMUP_ONLY=<named-subset>` only for startup diagnosis (empty performs the full production warm-up), and `P2P_FAKE_RUNOUT=1` only for fault-injection tests.

## Gates for every change

1. Run focused tests and `DEV=NULL python3 -m pytest test/null -q -n 16`; touched-file Ruff; `python3 -m mypy tinygrad/llm`; `git diff --check`. When launch env changes also run `DEV=NULL python3 -m pytest test/null/test_llm_service_launchers.py -q`.
2. Run all three full NULL production startups with `bash test/null/production_c2_warmup.sh`, `bash test/null/production_c5_warmup.sh` and `bash test/null/production_c9_warmup.sh`. C2 covers the Qwen text model, picture tower, cache allocation, JIT capture and run-again; its NULL stash defaults to 2 GB/device (override with `SEQ_STASH_GB`) so the test does not allocate the production 46 GB of real host RAM, with unchanged warm graph shapes. C5 covers the GLM Flash D34 model load and full sequence warm-up. C9 covers the GLM model load and full sequence warm-up. Require `NULLWARM: warm-up DONE`, `C5 NULLWARM: warm-up DONE` and `C9 NULLWARM: warm-up DONE`, respectively. Every production startup still requires its real release gates below; NULL results never prove real VRAM, collectives or outputs.
3. A model/kernel/scheduler change needs base/candidate **text-only greedy output or CHECK-hash identity** at 4k, 64k and 262k, and several users where scheduling matters. Indexer/candidate paths gate at >=12k. Qwen vision changes additionally need picture NLL/known-answer gates while text stays identical.
4. Numerics changes require predeclared paired teacher-forced NLL/top-1/KL against llama.cpp on identical GGUF ids and real code/agent windows; GLM also runs its C9 NLL/retrieval/GSM8K suite. Never hide a numerics change inside cleanup.
5. Final candidate real gates: the Qwen engine owner runs a six-card C2 start, effective-env and memory check, warm/cached/multiuser turns and Qwen text/picture outputs. the GLM Flash owner runs the six-card C5/D34 start, effective-env/memory check, native tool-call and reference checks, 18-concurrent-user turns and a 250K-token prompt; the GLM engine owner runs the 14-card C9 pair, effective-env/memory check, cached/multiuser turns and quality checks. Record commit, machine, command and raw output. NULL never proves real VRAM, collectives or performance; speed claims require a same-box same-hour back-to-back pair.
6. Deploy only after every service's required owner accepts the exact candidate. Retain preceding passing tree and weight manifest for rollback. Any newly-added environment knob must be documented here and fit contract/knob/short-diagnostic policy; the target is fewer than 80 names total.
