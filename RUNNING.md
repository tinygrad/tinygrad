# Running the three production inference services

Supported baseline: `redqwenflash` at `f3f31afb6c5268640d1602f371b406321c3732a9`, including the delivery commits accompanying this document. Production deployments must record the exact commit in `DEPLOYED_COMMIT`. There are three product configurations:

| Service | Model | Hardware | Context | Devices |
|---|---|---|---:|---|
| Qwen C2 | Qwen3.8 Flash Next UD-IQ4_XS **V3.1** + separate MTP + picture tower | 6x AMD 7900 XTX (RDNA3) | 262144 | `DEV=AMD`, 6 shards |
| GLM Flash C5 | GLM-5.3 Flash UD-IQ3_XXS, D34 quality configuration + mandatory picture tower | 6x AMD 7900 XTX (RDNA3) | 18 seats, up to 262144 tokens each from a 3,916,800-token shared pool (14 full-length at once); 524288 server cap | `DEV=AMD`, 6 shards |
| GLM C9 | GLM-5.3 UD-IQ4_XS; **text-only, no vision tower** | 14x AMD R9700 (RDNA4), ts5 | 262144 / 1048576 (1M profile: a 1,020,639-token request served with sane text at 170 W; 3 h 52 min soak at 180 W on 78b57b743; the current line (SEQ_KV4_FAST, chunk 4032): 1,020,640 tokens, first token 705.9 s at 180 W = results/w-pf53/ts5-acceptance.md FINAL-2) | `DEV=PCI+AMD`, `VFIO=1`, 14 shards |

**Vision is mandatory on Qwen C2 and GLM Flash C5, whose models support it.** Their launchers verify the picture file and its hash before start; the CLI also checks the loaded model architecture and rejects a missing/empty working image path. **C9 has no compatible vision tower and is explicitly text-only per the 12:00 decision.** Its launcher always clears `VISION_MMPROJ`; this is intentional, not an unset production option. On C2/C5, no-projector launches remain strictly limited to the offline test configurations, never a deployed service.

Only a machine's owner starts/stops service and only one GPU workload may use a box. Do not load the `amdgpu` kernel driver and do not run `modprobe`/DKMS: tinygrad's AM support drives the hardware. GPU jobs on shared infrastructure use its queue/ownership protocol. The three production launchers are `extra/llm_service/qwen_6x7900xtx.sh`, `glm_flash_6x7900xtx.sh` and `glm_14xr9700.sh` in this repository. VM/tunnel/proxy infrastructure is separate.

## Preconditions

1. Python prerequisites, on every box before its first start: Python 3 with **numpy** (the model loader), **jinja2** (the chat template) and, on the vision lines C2 / C5, **pillow >= 10.0.1** (Ubuntu 24.04: `sudo apt-get install python3-numpy python3-jinja2 python3-pil`; an iris 6x7900 XTX VM: the company's `tools/vm/setup.sh <vm>` once after its creation). The three launchers check these imports, then the weights' hashes, before any GPU is opened (`<launcher> check-weights` runs both checks alone): a missing module found after the cards were set up costs a mode1 reset in a VM. Use an immutable checkout; keep weights read-only.
2. Record a writable tinygrad cache (`XDG_CACHE_HOME`), persistent state/log directory, and ownership of the box. Never expose `SESSION_END_TOKEN` or API keys in argv, commits or logs. The HTTP service is loopback HTTP; TLS and external access belong to the front proxy.
3. Check all expected cards and free device/host memory. R9700 limit: **180 W a card on ts5 (GLM 5.3, 14 cards: the board's setting Nov 15, below); the former launcher default 210 W is NOT safe. `GLM53_POWER_LIMIT` is required (the launcher has no numeric default).** Qwen ran at **250 W/card** and GLM Flash D34 at **225 W/card**; use **225 W for Qwen on a hot host**, keep every hotspot **under 100 C**, and log it. A dropped/hung card is an incident, not a reset loop.

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

The mandatory GLM Flash picture tower uses `mmproj-F16.gguf` from the published `unsloth/GLM-5.3-Flash-GGUF` main branch: **1,128,047,296 bytes**, SHA-256 `922b809664d6a89cc1e40bfeaa63618000edc6651baf4082ce0718a784adf947`. A fresh picture VM copies it to `/home/runner/glmvis/mmproj-F16.gguf` (it is not under `/weights`); a different installation path is valid only if that byte content matches. Verify it before every production start.

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

**Every start ends with the customer smoke, through the public port, before the server is called serving** (exit code 0 =
`SMOKE PASSED`; ~1 h on a loaded box, mostly part c):

```bash
python3 extra/llm_service/smoke.py http://<public host:port> --slots <SEQ_SLOTS> --rows-4k <rows>
```

`--rows-4k` = the rows of a 3840x2160 picture on that line (picture-limit table below): Qwen and GLM Flash at 1,024 rows 1,008, Qwen at 8,192 8,160,
GLM Flash at 8,000 7,973; without it the count is printed, not checked. GLM Flash always thinks (its template opens `<think>`;
`enable_thinking` does nothing): add `--ret-max-tokens 1024 --draw-prompt "Do not think about this at all; your thinking must start with one
random fruit name. Which fruit?"` (part b then samples the reasoning's first token, which has a spread there).

It checks (a) 20 default-sampling chats (no stray-script characters, reasoning in `reasoning_content`; read the five it prints),
(b) top_p honoured and top_k / unknown fields accepted, (c) max(80, 1.25 x slots) conversations each return to their own secret, text and
one picture each, (d) 10 pictures in one request and a 3840x2160 picture's rows. Add `--no-pictures` on C9 (text-only). It cannot check
that the start used the launcher line printed here: the operator states that.

### Qwen C2, on its six-card serving VM

Work from the immutable tinygrad checkout. This is the full environment from the running V3.1 server (with `AM_VM_RESET=1` only on its first controlled start) and complete engine invocation:

```bash
setsid env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. PYTHONUNBUFFERED=1 HOME="$HOME" \
  DEV=AMD AM_MEC_PIPES=2 AM_IDLE_MS=1000 AM_POWER_LIMIT=250 \
  SPEC_DRAFTS=2 QSA_KV_BITS4_LAYERS=39,43,47,48 \
  SEQ_ROWS=4310016 SEQ_SLOTS=64 SEQ_SIZES=1,8,16,24,30 SEQ_MAX_ACTIVE=30 SEQ_STASH_GB=46 \
  SEQ_CKPT2=1 SEQ_BASE_ALIGN=24 \
  PREFILL_CHUNK=2304 PREFILL_SIZES=1152 PREFILL_PARTS=2 PREFILL_MB1_ROWS=1200 PREFILL_PK_CTX=131072 \
  PARALLEL=40 TOP_K=20 \
  CLOCK_FAST=2000 CLOCK_MANY=1900 CLOCK_PREFILL=2000 \
  VISION_MMPROJ=/weights/Qwen3.8-Flash-Next-UD-IQ4_XS-V31/mmproj/mmproj-F16.gguf \
  VISION_SLICE_MS=200 VISION_MAX_IMAGES=1024 VISION_CACHE_MB=1024 VISION_URL_CACHE_MB=512 \
  XDG_CACHE_HOME=/home/runner/vm-cache \
  python3 -m tinygrad.llm \
  --model /weights/Qwen3.8-Flash-Next-UD-IQ4_XS-V31/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf \
  --shard 6 --max_context 262144 --serve 8000 > qwen-c2.log 2>&1 & echo $! > qwen-c2.pid
```

Use 225 W on a hot host. Omit `AM_VM_RESET` for every ordinary start; on the first verified-clean driver start only, prepend `AM_VM_RESET=1`. Set `SESSION_END_TOKEN` separately in the process environment only if the admin endpoint is needed. `GET /v1/models` must identify Qwen and `GET /status` must report slots=64/max_active=30 with eight-bit-plus-late-four-bit cache capacity. Stop: `kill -TERM "$(cat qwen-c2.pid)"`; await process exit and release of all six cards. The equivalent supported wrapper is `extra/llm_service/qwen_6x7900xtx.sh start`, whose `status`/`stop` actions use its state dir.

### GLM Flash C5, on its six-card serving VM

D34 serves 18 concurrent users, each up to 262144 tokens (a 524288 server cap), Q4 KV, 18 warmed slots/active requests, 3,916,800 pooled rows and zero stash. THE LAUNCHER'S DEFAULT IS NOW THE ONE GLM PRODUCTION LINE (Nov 15, 22:3x): D34 plus the qualified host stash (`SEQ_STASH_GB=12 SEQ_STASH_ENTRIES=64`) plus the picture limits a Pokemon window needs (`VISION_MAX_IMAGES=2048 VISION_CACHE_MB=2048 VISION_URL_CACHE_MB=512 VISION_SLICE_MS=200 SERVE_RENDER_CACHE=1 VISION_BUCKETS=320,576`): a box started by the launcher never refuses a 9th picture (the engine's own default, 8, is for bare `python3 -m tinygrad.llm` runs). `SEQ_STASH_GB=0` gives back the benchmark's D34 exactly. The pool is shared: it holds 14 full 262,144-token conversations at once (3,916,800 / 262,144 = 14.9; 18 full ones need 4,718,592), so 18 seats fit only while the conversations average under 217,600 tokens; past that the least recent is evicted (and re-prefilled when it returns) or, with the stash profile below, parked in host RAM. The canonical complete environment is printed by the launcher without touching a GPU:

```bash
bash extra/llm_service/glm_flash_6x7900xtx.sh --print-command
bash extra/llm_service/glm_flash_6x7900xtx.sh start
```

The fixed engine invocation is `python3 -m tinygrad.llm --model /weights/GLM-5.3-Flash-UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf --shard 6 --max_context 524288 --serve 8000`. D34 ran at 225 W/card. Leave `SPEC_DRAFTS` and `SPEC_DRAFTS_BY_SIZE` unset: D34 inherits two drafts. Use `AM_FIRST_START=1` only for the first controlled start, not ordinary restarts. `status` and `stop` use the launcher's state dir. Every C5 start requires `VISION_MMPROJ=/home/runner/glmvis/mmproj-F16.gguf` (or the byte-identical file at another installation path), with unchanged capacity `SEQ_ROWS=3916800` for 18 slots:

```bash
VISION_MMPROJ=/home/runner/glmvis/mmproj-F16.gguf extra/llm_service/glm_flash_6x7900xtx.sh start
```

The C5 launcher's default line (since da35e3273, Nov 15) carries the picture limits itself: `VISION_SLICE_MS=200`,
`VISION_MAX_IMAGES=2048`, `VISION_CACHE_MB=2048`, `VISION_URL_CACHE_MB=512`, `SERVE_RENDER_CACHE=1` and `VISION_BUCKETS=320,576` (`VISION_MAX_TOKENS` stays at the engine's 1024); each can be
overridden by setting it before `start` (`--print-command` shows the resolved values). The benchmark runs before Nov 15 used the older
baseline (`VISION_MMPROJ` and `VISION_BUCKETS=576` only, the rest at engine defaults). `VISION_LATE` and `VISION_KERNELS` no longer select runtime behavior: both are folded into the active production path; independent relink/run-again remains. Mandatory tower memory/warm-up does not change model rows or draft counts. Flash's solo evidence establishes this launch configuration and picture availability, not a new multiuser or exactness pass.

With `SESSION_END_TOKEN` unset (the default), explicit backend session-end calls return the expected **403 `backend_session_end_disabled`**, not generation failures. An integrated proxy releases its own session bookkeeping, while backend conversations remain in bounded 18-slot residency until reuse, NONDROP eviction or restart; `ended_dropped=0` only counts processed explicit ends and is **not** proof of backend memory release. Actual coordinated cleanup requires a matching out-of-band token on proxy and backends plus active/completed/duplicate-end tests (including resident/stash drops, `already_ended` and later admissions); never expose the secret in `--print-command`, and enabling the endpoint alone does not move model-seat release before verifier execution.

**C5 stash profile (36 conversations on 18 seats).** `SEQ_STASH_GB=12 SEQ_STASH_ENTRIES=64 extra/llm_service/glm_flash_6x7900xtx.sh start`
adds a host-RAM stash: a conversation pushed out of the 18 device slots parks its states and cache in host memory and comes back without
a re-prefill. Sizes (a NULL C5 start; the company repo's stash qualification, Nov 15): a card holds 57.9 MB an entry and 1.97 MB a
3,072-token chunk; 12 GB / 64 entries = 4,205 chunks a card = 12,917,760 tokens beside the 3,916,800 pooled rows (64 parked
conversations of ~200K, 49 at a full 262K). The stash is allocated at warm-up: x6 cards = ~72 GB of host RAM (guest MemAvailable
94.5 -> ~23 GB on a 109 GB VM; leave it at least 8 GB). The launcher refuses a `SEQ_STASH_GB` too small for its entries (it would
silently hold no chunk). Conversations under `SEQ_STASH_CTX` (32,768 tokens) are not parked; they re-prefill. Proof (g1, tree
03896ef29, the company repo's stash qualification): greedy 4K / 64K / 262K equal to the reference; a 100K, a 262K and a 40K-with-pictures conversation
parked by 21 others resumed with the same 64-token greedy continuation as a never-parked twin and 0 tokens re-prefilled; 36
conversations on 18 seats for 33.7 min: 690 requests, 0 errors, 0 lost, every return from cache or stash. Its cost is the wait at
the seat: a cold first prompt of 34-90K tokens queues behind returning turns (all 36 first replies took 32 min). The proxy carries
it as one backend of 18 running / 36 resident / 36 live with a client timeout of at least 3,600 s: in that run the cold first replies took 109-1,933 s (median 570 s) and 2 of 36 took longer than 1,800 s (1,875 s and 1,933 s, cold prompts of 116K and 80K tokens waiting behind returning turns); 3,600 s is that maximum with a margin of about 2x, because the wait grows with the number of returning conversations ahead of a cold one.

**Picture size (both six-card models).** By default a picture is downscaled to at most 1,024 tower rows (`VISION_MAX_TOKENS`, about
0.8 MP for GLM's 28-pixel rows, 1 MP for Qwen's 32-pixel rows); the publishers allow 8,000 (GLM, `max_image_tokens`) and 16,384 (Qwen).
A higher cap keeps a bigger tower bucket's buffers on every card and is paid in resident cache rows; the launchers set the measured pair
when only `VISION_MAX_TOKENS` is given and refuse any other cap without `SEQ_ROWS`:
`VISION_MAX_TOKENS=8000` (GLM) -> `SEQ_ROWS=1505280` (card 0 ~0.15 GB free after warm-up; tower ~2 s at 1,768 rows, ~5.5 s at 4,070,
~15 s at 7,973); `VISION_MAX_TOKENS=8192` (Qwen) -> `SEQ_ROWS=3299328` (tower ~10 s for a 4K picture). Proof (Nov 15): GLM at 8,000 rows:
text greedy 4K / 64K / 262K equal; 13 pictures up to 3840x2160 (7,973 rows, a 67x119 grid) within 0.02 nats of llama.cpp's
log-likelihood; Qwen at 8,192: greedy 4K / 64K equal to the production boots, 4K pictures described right. Qwen's 16,384 needs about
4 GB more a card (~2.2M rows): not run. Fewer resident rows serve the same seats through the host stash (parks and restores), at a
throughput cost that was not measured; the benchmark numbers in this file were measured at 1,024.

**Which box gets which limit, and what a customer is told.** A picture over the limit is DOWNSCALED, aspect kept, never refused
(refused, with a 400 that says why, only when the file decodes to more than 40 MP, `VISION_MAX_PIXELS`; a large JPEG is first decoded at 1/2-1/8 scale); the response's `usage.prompt_tokens` counts the rows the
model saw. Whoever offers a box to customers publishes its line of this table with it:

| Box role | Start line | Limit | A 3840x2160 (4K) picture is seen at | 1920x1080 | 4032x3024 (12 MP phone photo) |
|---|---|---|---|---|---|
| Coding (DeepSWE, SWE-bench Pro proxies), GLM Flash | C5 benchmark line | 1,024 rows | 1176x661 (shrunk) | 1176x661 (shrunk) | 1008x756 (shrunk) |
| Coding, Qwen | C2 benchmark line | 1,024 rows | 1344x768 (shrunk) | 1344x768 (shrunk) | 1152x864 (shrunk) |
| Pictures (vision work, photo customers), GLM Flash | C5 + `VISION_MAX_TOKENS=8000` | 8,000 rows (publisher's) | 3332x1874 (shrunk, as the publisher's own processor) | full size | 2874x2156 (shrunk) |
| Pictures, Qwen | C2 + `VISION_MAX_TOKENS=8192` | 8,192 rows | full size (3840x2176 canvas) | full size | 3328x2496 (shrunk) |
| Pokemon (320x288 screenshots), both | the Pokemon lines below | 1,024 rows | (not its load) | | |

(Sizes from `qwen_vision.tower_size`: GLM rows are 28 x 28 pixels, Qwen's 32 x 32. Qwen's publisher allows 16,384 rows = 4032x3024
at full size; not run here, see above.) So: a coding box shrinks every picture larger than about one megapixel and says so only
through this table and `usage.prompt_tokens`; send photos to a picture box. The server does not report a resize in the response.

### GLM C9, on ts5 — text-only; no vision tower

**GLM-5.3 is text-only under the 12:00 product decision.** It has no compatible vision encoder or projector. C9's launcher forces `VISION_MMPROJ` empty, including when the parent shell supplied another value; the generic CLI requires a projector only for the supported Qwen and GLM Flash vision models. Do not substitute GLM Flash's incompatible tower or claim that a file gives C9 vision.

The nine model-shard hashes above are the complete service input. `--print-command` requires no image file and is safe for review. Set the power value only from the completed power qualification; the launcher deliberately has no numeric default:

```bash
bash extra/llm_service/glm_14xr9700.sh --print-command
GLM53_POWER_LIMIT=180 AM_PERF_LEVEL=high CLOCK_PREFILL=GFXCLK=2300 \
  bash extra/llm_service/glm_14xr9700.sh start
```

That power gate and the owner-only host controls remain unchanged. This documentation authorizes no launch and makes no hardware, memory, quality or multiuser claim.

**Profiles stay independent of vision policy.** The default is half-cache. The explicit 1M candidate command is:

```bash
GLM53_POWER_LIMIT=180 AM_PERF_LEVEL=high CLOCK_PREFILL=GFXCLK=2300 \
  GLM_KV=q4 MAX_CONTEXT=1048576 SEQ_ROWS=1048576 SEQ_SLOTS=16 SEQ_MAX_ACTIVE=16 \
  P2P_GLM5_OFF=1 FREQS_HOST=1 AMD_SCRATCH_MAX=64 SEQ_COPY_SCRATCH=1 \
  PREFILL_CHUNK=4032 PREFILL_SIZES=1344,2688 SEQ_KV4_FAST=1 \
  bash extra/llm_service/glm_14xr9700.sh start
```

`GLM53_POWER_LIMIT=180` = the ts5 power setting the board committed to (Nov 15, ~17:05 PST: "commit to 180W and make it as fast as we can there"), the top rung of w-pf53's uniform ladder uladder-1 (all five rungs 140-180 W passed by events: no traceback / 5xx / p2p run-out / link drop). At 180 W the speed equals 170 W (results/w-pf53/tune180.md: cards average ~133 W, power is not the limit; what the 1M profile pays is its q4 cache); the 8-card supply shows half-second peaks above its 1,700 W rating (2-4 samples a 12-min run; in soak-4 (3 h 52 min, the PREFILL_CHUNK=2688 / no-FAST line) 22 of 12,162 samples in its first 1 h 44 min, highest 2,361 W; its 10-s means stay under 1,450 W). This line (FINAL-2: `SEQ_KV4_FAST=1` + `PREFILL_CHUNK=4032`) ran 30 minutes of mixed load (196/196 ok, 0 tracebacks / run-outs / AER) + one 1,020,640-token request (TTFT 705.9 s, 53.4 tok/s) clean (FINAL-2, results/w-pf53/ts5-acceptance.md); the 3 h 52 min soak (soak-4) was on the PREFILL_CHUNK=2688 / no-FAST line; 4032 changes greedy texts vs 2688 (other chunk boundaries), as the 262k config already does (`SEQ_KV4_FAST` alone: the same bits). Rows on this line: first token 2.20 / 31.67 / 146.76 s at 4k / 64k / 262k, decode 96.7 / 92.4 / 89.1 tok/s, 41.7 / 25.9 / 14.3 tok/s a user at 4 / 8 / 16 users. The clock knobs are part of that power setting (decode uncapped, prefill capped at 2300 MHz, perf level high); the launcher forwards them but sets none, so they belong on the start line. `SEQ_COPY_SCRATCH=1` keeps ~0.2-0.5 GB a card of the move jits' own arenas free (the 1M start runs with ~0.48 GB a card free with it). One 1M conversation at a time beside short ones; a second 1M request waits. `AM_FIRST_START=1` only on the first start after a reboot.

The launcher expands q4 to `SEQ_FP8=1`, `SEQ_FP8_IDX=0`, `SEQ_KV4=1`; `GLM_KV=fp8` is a separate mode. Real 14-card start, capacity and correctness checks remain required for both configurations.

## Pokemon (one game through a whole 262,144-token window, screenshots in every turn)

A game is a workflow on the normal C2 / C5 servers with a few settings changed; vision stays mandatory. One fresh six-card 7900 XTX VM,
**one server start per boot**; a changed setting needs a fresh VM. Speed and parcel counts: `results/pokemon/SPEED.md` and `PARCEL.md`
in the company repo (final rows 21:00 Nov 15).

**Qwen server** (the C2 launcher with six overrides; the game sends `preserve_thinking=false`, so `SEQ_CKPT_TAIL=1` is required:
without it every turn re-reads the whole conversation from token 0):

```bash
SPEC_DRAFTS=3 PREFILL_SIZES=192,384,1152 VISION_MAX_IMAGES=2048 VISION_CACHE_MB=2048 SERVE_RENDER_CACHE=1 SEQ_CKPT_TAIL=1 \
  VISION_MAX_BODY_MB=64 VISION_URL_CACHE_MB=4608 SEQ_STASH_GB=41 bash extra/llm_service/qwen_6x7900xtx.sh start
```

(The last three since Nov 16 02:4x, g4: the game resends every screenshot each turn, ~33.6 MB at ~840 turns, over the 32 MB default;
the URL cache keeps every decoded screenshot of 12 games, paid from the host stash, 46 -> 41 GB.)

**GLM Flash server** (the C5 launcher: since Nov 15 22:3x its default line carries all of the values below; the line is kept as the record of the R2 games; `VISION_BUCKETS=320,576` adds a
320-patch bucket for the game's 240x216 screenshot (72 rows = 288 patches; C5's own 576 alone pads it to 576); the GLM R2 games ran this line):

```bash
VISION_MMPROJ=/home/runner/glmvis/mmproj-F16.gguf VISION_SLICE_MS=200 VISION_MAX_TOKENS=1024 VISION_MAX_IMAGES=2048 VISION_CACHE_MB=2048 \
  VISION_URL_CACHE_MB=512 VISION_BUCKETS=320,576 SERVE_RENDER_CACHE=1 AM_POWER_LIMIT=250 bash extra/llm_service/glm_flash_6x7900xtx.sh start
```

The GLM arm of Nov 16 (g1, 8 games at 480x432 + 8 at 360x324) adds `VISION_MAX_BODY_MB=64 VISION_URL_CACHE_MB=7168 SEQ_STASH_GB=6`:
a decoded screenshot is 677,376 B at 480x432 and 366,912 B at 360x324, 16 games x 800 = 6.7 GB, paid from the stash (16 games fit the
18 slots; with many other conversations the smaller stash makes a new one wait behind 180-s sticky ones).
A game that resends every screenshot of the window in each request needs a body cap above the 32 MB default once its screenshots are
larger than 240x216: 842 PNG screenshots at 360x324 (~38.9 KB each) are ~32.7 MB, a 413. For those add `VISION_MAX_BODY_MB=64` to the
line (both launchers pass it; the Qwen Pokemon line on g4 runs with 64 since Nov 16 02:4x).

**Qwen clients that drop old reasoning** in general (`chat_template_kwargs: {preserve_thinking: false}`): the same `SEQ_CKPT_TAIL=1`.
A client that sends nothing gets `preserve_thinking=true` and reuses its cache without it. Default off.

**The game harness** (teeny, `~/fun/pi_plays_pokemon` at `104cacd`): prompt "Beat the game." plus the seven sentences of arm R3
(`.pi/RULES_R3.md`, below: R2's six + the board's seventh, Nov 15); no history rewrite (every screenshot and message stays); the run ends at the window's compaction point.

```bash
cd ~/fun/pi_plays_pokemon
# Qwen
EMU_TEXT_SETTLE=1 PI_ACT_SEQ=1 PI_IMAGE_SIZE=240x216 PI_PRESERVE_THINKING=0 PI_RULES=R3 \
PI_QWEN_BASE_URL=http://127.0.0.1:<qwen tunnel>/v1 ROM_PATH=$PWD/roms/pokemon_red.gb PI_FRESH=1 PI_IMAGE_SCALE=2 PI_IMAGE_REFS=1 \
PI_TEMPERATURE=1.0 EMU_PORT=8900 EMU_URL=http://127.0.0.1:8900 \
.venv/bin/python campaign.py --full-context --block-minutes 900 --run qwen38-flash-full-NNN --change "<label>"
# GLM Flash
PI_IMAGE_SIZE=240x216 PI_RULES=R3 \
PI_GLM_BASE_URL=http://127.0.0.1:<glm tunnel>/v1 ROM_PATH=$PWD/roms/pokemon_red.gb PI_FRESH=1 PI_IMAGE_SCALE=2 PI_IMAGE_REFS=1 \
PI_TEMPERATURE=1.0 PI_TOP_P=0.95 EMU_PORT=8901 EMU_URL=http://127.0.0.1:8901 \
.venv/bin/python campaign.py --full-context --block-minutes 900 --provider local-glm-flash --model "GLM 5.3 Flash" --run glm53-flash-full-NNN --change "<label>"
```

The seven R3 sentences (picture references on for both models: `PI_IMAGE_REFS=1`, the server keeps pictures by hash):
- In a new town, enter every building that has a door before you leave the town.
- When a person asks you to do something, or gives you something to take to someone, do that next, before you explore further.
- If a person blocks your way and talking to them does not open it, something elsewhere must be done first: go back and finish what you have not finished.
- When you are given something to take to someone, go back the way you came to where you met that person, and talk to them.
- If the screen keeps coming back to the same few views, what you are doing does not work: try a wall, an edge or a direction you have not tried yet.
- Read to the end everything a person says: it often says where to go next.
- To leave a room, walk onto its stairs, door or mat and keep pressing the same direction until the screen changes.

## Benchmarks (source: the company repo's results/phase4/benchmarks.md, copied 21:34 Nov 15)

| Configuration | Benchmark | Resolved / valid | Accuracy (95% interval) | Published | Serving tree | Seats a box |
|---|---|---:|---|---:|---|---:|
| Qwen C2 (6x 7900 XTX, 262,144 a user) | DeepSWE v1.1 | 656 / 1,130 (ten runs of 113) | 58.05% (55.15-60.90) | 58.7% | 774202f33, 5de5e797f | 30 |
| GLM Flash C5 (6x 7900 XTX, 262,144 a user) | DeepSWE v1.1 | 276 / 452 (four runs) | 61.06% (56.49-65.45) | 63.4% | 542faaa80 .. 5ce1f87eb | 18 |
| Qwen C2 | SWE-bench Pro (642 public tasks) | 618 / 642 | 96.26% (94.50-97.48) | ~62% (other harness; the gap is not explained) | 5ce1f87eb | 30 |
| GLM Flash C5 | SWE-bench Pro | pending at 21:34 Nov 15 | | | | 18 |

The benchmark runs were served by the trees named, not by this branch's head: this branch's proof is the identity gates
(greedy token-for-token equality at 4K / 64K / 262K and picture NLL against llama.cpp within 0.02 nats).
GLM 5.3 C9 (ts5): its bars against tinyamd1 are in the company repo, results/w-pf53/ts5-acceptance.md.

## What each configuration cannot do

**All three:** one server process a box and one start per VM boot; no second model beside it. The context is per conversation; a
conversation longer than its window is refused, not truncated. Sampling is the model's own (temperature, top-p, top-k 20 for Qwen); no
grammar-constrained decoding beyond the server's JSON `response_format` unfencing.

**Qwen C2 and GLM Flash C5, pictures:** only `data:` URLs (base64) in `image_url` parts; an `http(s)` URL, a picture in a system message, a
file over 20 MB or over 40 megapixels decoded, or more pictures a request than `VISION_MAX_IMAGES` (C2 1024, Pokemon 2048) is refused with a
400 and its reason. A picture is resized to at most 1,024 rows of the model (about one megapixel); larger pictures lose detail, they are
not tiled. No video, no audio.

**Qwen C2:** 30 conversations a box at 262,144 tokens each. DeepSWE v1.1 58.05% (published 58.7%) and SWE-bench Pro: the table below.
One request decodes at roughly 40 tokens a second on a full box (30 users), more on an idle one; not the 60 the board asked for.

**GLM Flash C5:** 18 conversations a box, each up to 262,144 tokens from the shared 3,916,800-token pool (14 at full length at once; the server's cap 524,288 is for scheduling, not one longer conversation).
Two draft tokens a step (three measured slower for Pokemon).

**GLM C9 (ts5):** text only: GLM-5.3 has no vision tower. A request with a picture is NOT rejected: it is answered 200 with a text reply
that the server cannot see images (the publisher's behaviour for a text-only model; w-pf53's smoke on ee9eee4a8, Nov 16). A client that
needs an error must check for pictures itself. The 1M-token profile serves ONE 1M conversation at a
time beside short ones (a second one waits); its first token at ~1M context takes ~12 minutes (705.9 s at 180 W on the FINAL-2 line; 739 s at 170 W and 885 s at 120 W on older lines). Its power setting is
180 W a card (the board's choice), not the cards' maximum; the bars against tinyamd1 are in the company repo `results/w-pf53/ts5-acceptance.md`.
A finished conversation keeps its slot 30 s (the launcher's SEQ_STICKY=30; with 180 s and no stash, 16 short chats in 3 minutes made
the 17th new one wait 101 s, Nov 16) for its next turn: an agent's tool call longer than that may come back to a re-prefill.
Sampling: temperature and top_p (default the GGUF's: temperature 1.0, top_p 0.95); **no top_k**: a request's `top_k` (and a server
`TOP_K`) is ignored, one log line `top_k ... ignored` (glm_dsa's sampler has none). Before p3/w-pf53-sampfix (Nov 16) top_p was ignored
too on every decoding step (whole-vocabulary sampling: stray foreign tokens; company repo results/incidents/ts5-stray-tokens/).

## API and production environment contract (72 names)

API sampling: a request without a temperature uses the model's GGUF `general.sampling.temp` and `top_p` (Qwen 3.8 Flash: 1.0 / 0.95); absent sampling metadata keeps the previous greedy behavior, explicit request values are unchanged, and startup logs report the defaults.
**Penalties are accepted and IGNORED** (every model, every line): a request's `presence_penalty`, `frequency_penalty` and
`repetition_penalty` (any value) get HTTP 200, no error and no log line, and the reply is sampled with temperature / top_p / top_k
only. Qwen's publisher recommends presence_penalty 0 (thinking) / 1.5 (non-thinking) against endless repetition: a client that
relies on it gets none here (Pokemon, Nov 16: 0.2% of Qwen replies looped one sentence to the 32,768-token cap). `smoke.py` part b
sends all three and prints this.

Count: **30 launch options, 30 operator knobs and 12 short diagnostics (72 total)**. Kernel/model/scheduler decisions already folded into the implementation are not exposed as launch assignments; the launchers and NULL harnesses no longer pass those obsolete names.

Any setting not listed here is not supported: its production value is compiled into the tree. Values below are launch defaults/current validated profiles, not permission to run arbitrary combinations. Capacity, sharding and cache-format changes require their own real-memory and text gates. Empty means disabled/unset.

### Shared C2/C5/C9 hardware/model launcher contract (21)

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
| `SEQ_GATHER_UNION` | default | default | 65536 | Position enabling sparse union gathers. |
| `SEQ_MAX_ACTIVE` | 30 | 18 | 32 | Concurrent admission cap. |
| `SEQ_ROWS` | 4310016 (example) | 3916800 | 268800 | Reserved cache rows for this capacity profile. |
| `SEQ_SIZES` | 1,8,16,24,30 | 1,2,4,8,12,13,18 | 1,4,8,16 | Warmed decode batch sizes. |
| `SEQ_SLOTS` | 64 | 18 | 32 | Warmed resident conversation slots. |
| `SEQ_STASH_ENTRIES` | derived from stash profile; unset in the running env | 128 (stash profile: 64) | 128 | Host-stash entry cap; with `SEQ_STASH_GB` > 0 each entry takes ~57.9 MB a card before any chunk. |
| `SEQ_STASH_GB` | 46 | 0 (stash profile: 12) | 0 | Host-stash bytes per device. |
| `SPEC_DRAFTS` | 2 | unset (model default 2) | 2 | Speculative tokens except per-size override. |
| `SPEC_DRAFTS_BY_SIZE` | empty | empty | 1:2,2:2,4:3,8:3,16:3 | Draft count by active batch size. |
| `VFIO` | auto/unset | auto/unset | 1 | Device pass-through mode. |
| `AM_VM_RESET` | controlled first start only | controlled first start only | controlled first start only | Permission to reset AM hardware once; ordinary starts forbid it. |
| `VISION_MMPROJ` | required, verified nonempty file | required, verified nonempty file | unsupported and forced empty | Picture-projection GGUF on C2/C5; intentionally not present on C9. |

### Additional C5/D34 hardware/model contracts (9)

| Name | D34 value | Operator changes |
|---|---|---|
| `BATCH_SLOTS` | 8 | Scheduler batch-slot capacity on this RDNA3 profile. |
| `LOAD_DEFER` | 1 | Deferred weight-load schedule used by these GLM profiles. |
| `AMD_SCRATCH_MIN` | 576 | Minimum AM scratch reservation on this profile. |
| `AMD_SCRATCH_MAX` | 64 | Maximum AM scratch reservation on this profile. |
| `SEQ_COPY_SCRATCH` | 1 | Scratch capacity for host/device sequence copies. |
| `SEQ_IDX_SHARD` | 1 | Shard the D34 attention index over devices. |
| `SEQ_KV4` | 1 | Four-bit latent-KV cache format selected by D34. |
| `PREFILL_SLICE` | 1536 | Rows allowed in one scheduled prefill slice. |
| `SEQ_STARVE_ALT` | 1 | Alternative anti-starvation scheduling policy. |

### Operator knobs documented for operations (30)

| Name | Default/profile | Purpose |
|---|---|---|
| `API_KEY` | empty (open loopback) | Comma-separated Bearer keys; prefer file for real secrets. |
| `API_KEY_FILE` | unset | File containing one Bearer key per line. |
| `ENC_CACHE` | 256 | Maximum cached chat-template encodings. |
| `KEEPALIVE` | 15 s | SSE keepalive interval. |
| `MODEL_ALIASES` | C9 `zai-org/GLM-5.3` | Additional `/v1/models` ids; empty disables. |
| `MODEL_OWNED_BY` | tinygrad | Owner metadata in `/v1/models`. |
| `P2P_AR_TIMEOUT_MS` | C9 20000 | Abort stuck peer allreduce before host signal timeout. |
| `SEQ_CKPT2` | empty/off; validated C2 arms use 1 | Keep two prompt-end checkpoints a sequence. |
| `SEQ_PROTECT` | 60 s | Eviction protection for active conversations. |
| `SEQ_RESERVE` | 4096 rows | Cache rows excluded from admission. |
| `SEQ_STASH_CTX` | 32768 | Minimum context eligible for stash retention. |
| `SEQ_STICKY` | 180 s (C9 launcher: 30 s) | Idle-conversation residency grace period. |
| `SESSION_END_TOKEN` | required for admin endpoint | Bearer credential for session termination. |
| `AGING` | 0 | Per-second urgency used by the waiting-request scheduler; changing it affects convoy avoidance. |
| `SEQ_STARVE` | 600 seconds | Wait threshold granting starvation priority; use a measured workload-specific override such as 30 only explicitly. |
| `SEQ_CKPT_TAIL` | 0 (off) | Keep a prior-generation boundary checkpoint when clients discard old reasoning, preventing whole-conversation recompute. |
| `SERVE_RENDER_CACHE` | 1 in the C5 launcher's default; Qwen C2: off unless set (its Pokemon and picture lines above set 1) | Optional chat-template render cache; memoized renders can change server-side cache behavior, so enable only with output-identical API tests and expected cache controls. |
| `VISION_BUCKETS` | C5 launcher default 320,576 (older C5 benchmark runs: 576); unset on C2 | Additional picture-token tower buckets for frequent image sizes. Every extra bucket adds warm-up time and graph memory; run every bucket again and leave unset unless measured useful. |
| `VISION_CACHE_MB` | C2 launcher default 1024; C5 launcher default 2048 (older C5 benchmark runs: engine default) | Maximum host RAM for cached picture rows/profile features: this many MB stays resident when full, in addition to decoded pictures and engine state. Size a shared service from measured RSS, not either dedicated-box profile. |
| `VISION_URL_CACHE_MB` | C2 and C5 launcher default 512; Pokemon lines 4608 (Qwen) / 7168 (GLM, Nov 16) | Maximum host RAM for decoded pictures keyed by data-URL hash: up to this many MB avoids repeated decoding but remains resident beside the row cache and engine state. Reduce it for a shared service after measuring its working set. |
| `VISION_DECODERS` | 2 | Concurrent picture decode workers. |
| `VISION_DECODE_WAIT` | 30 s | Picture decode deadline. |
| `VISION_MAX_BODY_MB` | 32 | HTTP request body cap (413 above it); 64 for full-resend games with 360x324 screenshots. |
| `VISION_MAX_BYTES` | 20000000 | Encoded bytes allowed for one picture. |
| `VISION_MAX_IMAGES` | C2 launcher default 1024; C5 launcher default 2048 (older C5 benchmark runs: the engine's 8); Pokemon 2048; bare engine default 8, typical explicit limit 8–32 | Maximum pictures per request. Each can consume up to the byte/pixel/token caps plus decode, tower and queue work; the dedicated 1024 limit admits far more concurrent work and cache pressure. |
| `VISION_MAX_PIXELS` | 40000000 | Decoded pixel cap per picture. |
| `VISION_MAX_TOKENS` | 1024 | Picture token-row/tower memory cap. |
| `VISION_SLICE_MS` | 200 in validated picture services; generic default 300 | Target picture work per scheduler slice. |
| `WARMUP_JITS` | C5/D34 `all` | Named start-time JIT subset; `all` warms every supported production graph. |

### Short diagnostics list (12)

These are off unless diagnosing or validating. `SERVE_RENDER_CHECK=64` checks that many cached chat-template renders byte for byte; compare ordinary and cached modes before enabling `SERVE_RENDER_CACHE`. Keep those validation runs separate from the shared picture service.

These are off unless diagnosing or validating. Never enable fault injection in customer service. `DEBUG_LOAD`, `MEM_TRACE`, `AMD_SCRATCH_LOG`, `KPROF` (with `KPROF_TOP=30`), `LIN_NAMES`, `TELEMETRY_LOG=<jsonl-path>`, `SERVE_TRACE=<trace-path>`, `DISABLE_AMD_KERNELS=1` for generic-kernel reference runs, `RING_RDNA4=1` only on an owned RDNA4 validation job, `P2P_FAKE_RUNOUT=1` only for fault-injection tests.

## Gates for every change

1. Run focused tests and `DEV=NULL python3 -m pytest test/null -q -n 16`; touched-file Ruff; `python3 -m mypy tinygrad/llm`; `git diff --check`. When launch env changes also run `DEV=NULL python3 -m pytest test/null/test_llm_service_launchers.py -q`.
2. Run all three full NULL production startups with `bash test/null/production_c2_warmup.sh`, `bash test/null/production_c5_warmup.sh` and `bash test/null/production_c9_warmup.sh`. C2 covers the Qwen text model, picture tower, cache allocation, JIT capture and run-again; its NULL stash defaults to 2 GB/device (override with `SEQ_STASH_GB`) so the test does not allocate the production 46 GB of real host RAM, with unchanged warm graph shapes. C5 covers the GLM Flash D34 model load and full sequence warm-up. C9 covers the GLM model load and full sequence warm-up. Require `NULLWARM: warm-up DONE`, `C5 NULLWARM: warm-up DONE` and `C9 NULLWARM: warm-up DONE`, respectively. Every production startup still requires its real release gates below; NULL results never prove real VRAM, collectives or outputs.
3. A model/kernel/scheduler change needs base/candidate **text-only greedy output or CHECK-hash identity** at 4k, 64k and 262k, and several users where scheduling matters. Indexer/candidate paths gate at >=12k. Qwen vision changes additionally need picture NLL/known-answer gates while text stays identical.
4. Numerics changes require predeclared paired teacher-forced NLL/top-1/KL against llama.cpp on identical GGUF ids and real code/agent windows; GLM also runs its C9 NLL/retrieval/GSM8K suite. Never hide a numerics change inside cleanup.
5. Final candidate real gates: the Qwen engine owner runs a six-card C2 start, effective-env and memory check, warm/cached/multiuser turns and Qwen text/picture outputs. the GLM Flash owner runs the six-card C5/D34 start, effective-env/memory check, native tool-call and reference checks, 18-concurrent-user turns and a 250K-token prompt; the GLM engine owner runs the 14-card C9 pair, effective-env/memory check, cached/multiuser turns and quality checks. Record commit, machine, command and raw output. NULL never proves real VRAM, collectives or performance; speed claims require a same-box same-hour back-to-back pair.
6. Deploy only after every service's required owner accepts the exact candidate. Retain preceding passing tree and weight manifest for rollback. Any newly-added environment knob must be documented here and fit contract/knob/short-diagnostic policy; the target is fewer than 80 names total.
