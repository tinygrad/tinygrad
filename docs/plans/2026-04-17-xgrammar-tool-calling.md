# XGrammar Structured Outputs and Tool Calling Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add xgrammar-backed constrained decoding to tinygrad’s OpenAI-compatible LLM server, including structured outputs, auto/required tool choice, forced single-tool choice, and optional parallel tool calling, then verify it with unit/integration tests and a real Qwen3.5 2B run on macOS.

**Architecture:** Refactor generation so `tinygrad.llm.model.Transformer` can expose logits before token sampling. Build an optional xgrammar integration layer that compiles request-specific grammars from JSON schema and tool definitions using GGUF tokenizer metadata rather than HuggingFace tokenizers. Extend the OpenAI-compatible server to translate OpenAI request fields (`response_format`, `tools`, `tool_choice`, `parallel_tool_calls`) into constrained decoding settings and to emit OpenAI-shaped tool-call responses for both streaming and non-streaming paths.

**Tech Stack:** tinygrad LLM server, GGUF tokenizer metadata, optional `xgrammar` Python package, OpenAI Python client, unittest/pytest, Qwen3.5 2B GGUF on macOS.

---

### Task 1: Set up a Python 3.11 test environment with optional xgrammar dependency

**Objective:** Ensure the repo has a reproducible environment capable of running tinygrad LLM tests and xgrammar integration tests.

**Files:**
- Modify: `pyproject.toml`
- Create: `docs/plans/2026-04-17-xgrammar-tool-calling.md`

**Step 1: Write failing test expectation**

Add tests that import the optional xgrammar helper and skip cleanly if xgrammar is unavailable, so packaging behavior is verified.

**Step 2: Run test to verify failure**

Run: `python3.11 -m pytest test/null/test_llm_server.py -q`
Expected: FAIL after new imports/tests are added because helper/dependency wiring does not exist yet.

**Step 3: Write minimal implementation**

Add an optional dependency group in `pyproject.toml` for xgrammar-backed LLM testing and ensure runtime imports are lazy.

**Step 4: Run test to verify pass**

Run: `python3.11 -m pytest test/null/test_llm_server.py -q`
Expected: PASS or clean skips for xgrammar-specific tests.

**Step 5: Commit**

```bash
git add pyproject.toml docs/plans/2026-04-17-xgrammar-tool-calling.md
git commit -m "build: add xgrammar llm test extras"
```

### Task 2: Refactor `Transformer` generation to expose logits before sampling

**Objective:** Make constrained decoding possible without breaking cache reuse or portability across CPU/MPS/ROCm/CUDA/WebGPU backends.

**Files:**
- Modify: `tinygrad/llm/model.py`
- Test: `test/unit/test_llm_server.py`

**Step 1: Write failing tests**

Add tests asserting:
- a new logits-returning generation path exists
- unconstrained generation still works
- constrained selection can override the next token without breaking cache behavior

**Step 2: Run test to verify failure**

Run: `python3.11 -m pytest test/unit/test_llm_server.py -q`
Expected: FAIL because logits/sampler hooks do not exist.

**Step 3: Write minimal implementation**

Refactor `Transformer.forward`/`generate` so generation uses:
- prefill/cache logic as today
- a logits-producing call
- a pluggable token-selection callback for unconstrained and constrained modes

**Step 4: Run test to verify pass**

Run: `python3.11 -m pytest test/unit/test_llm_server.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add tinygrad/llm/model.py test/unit/test_llm_server.py
git commit -m "feat: expose llm logits for constrained decoding"
```

### Task 3: Add GGUF-tokenizer-to-xgrammar integration helpers

**Objective:** Create a reusable layer that converts tinygrad tokenizer/model metadata into xgrammar tokenizer/compiler state and request-specific grammars.

**Files:**
- Create: `tinygrad/llm/xgrammar_support.py`
- Modify: `tinygrad/llm/cli.py`
- Test: `test/unit/test_llm_server.py`

**Step 1: Write failing tests**

Add tests asserting the helper can:
- derive tokenizer info from `SimpleTokenizer` and GGUF vocab metadata
- create JSON-schema constrained decoders
- create tool-calling constrained decoders for `auto`, `required`, and forced single-tool modes
- preserve stop token ids and model vocab size semantics

**Step 2: Run test to verify failure**

Run: `python3.11 -m pytest test/unit/test_llm_server.py -q`
Expected: FAIL because helper module does not exist.

**Step 3: Write minimal implementation**

Implement:
- lazy xgrammar import helpers
- preset-to-vocab-type mapping for GGUF tokenizers
- compiler cache per loaded model/tokenizer
- grammar builders for JSON schema and OpenAI tool calling, including optional parallel calls

**Step 4: Run test to verify pass**

Run: `python3.11 -m pytest test/unit/test_llm_server.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add tinygrad/llm/xgrammar_support.py tinygrad/llm/cli.py test/unit/test_llm_server.py
git commit -m "feat: add xgrammar gguf integration"
```

### Task 4: Extend request parsing and response shaping for structured outputs and tool calls

**Objective:** Support OpenAI-compatible request/response fields for structured output and tool calling in streaming and non-streaming modes.

**Files:**
- Modify: `tinygrad/llm/cli.py`
- Test: `test/null/test_llm_server.py`

**Step 1: Write failing tests**

Add integration-style server tests covering:
- `response_format={type: json_schema}` non-streaming
- `tools` + `tool_choice="auto"`
- `tools` + `tool_choice="required"`
- `tools` + forced named tool
- `parallel_tool_calls=False` vs `True`
- streaming tool-call deltas
- assistant/tool history serialization for a follow-up request

**Step 2: Run test to verify failure**

Run: `python3.11 -m pytest test/null/test_llm_server.py -q`
Expected: FAIL because request fields/response shapes are unsupported.

**Step 3: Write minimal implementation**

Update `Handler` to:
- parse request constraints
- build prompt ids while supporting assistant/tool messages
- choose constrained vs unconstrained generation
- parse generated tool-call text into OpenAI response objects
- emit `finish_reason="tool_calls"` when appropriate
- stream `delta.tool_calls` for tool responses

**Step 4: Run test to verify pass**

Run: `python3.11 -m pytest test/null/test_llm_server.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add tinygrad/llm/cli.py test/null/test_llm_server.py
git commit -m "feat: add structured outputs and tool calling api support"
```

### Task 5: Add a Qwen3.5 2B model alias and real end-to-end smoke tests

**Objective:** Verify real constrained decoding with the requested Qwen3.5 2B model on macOS while keeping implementation portable to any tinygrad-supported backend.

**Files:**
- Modify: `tinygrad/llm/cli.py`
- Create: `test/external/external_test_llm_xgrammar.py`

**Step 1: Write failing tests**

Add external tests that:
- boot the tinygrad OpenAI server with Qwen3.5 2B GGUF
- request structured JSON output
- request one tool call
- request parallel tool calls when allowed

**Step 2: Run test to verify failure**

Run: `python3.11 -m pytest test/external/external_test_llm_xgrammar.py -q`
Expected: FAIL because alias/tests/server support do not yet exist.

**Step 3: Write minimal implementation**

Add a `qwen3.5:2b` model alias and any helper logic needed for real Qwen XML/tool-call formatting.

**Step 4: Run test to verify pass**

Run with a local backend on macOS first, then with any explicitly selected backend available on the machine.
Expected: PASS or clean skip when the model/runtime is unavailable.

**Step 5: Commit**

```bash
git add tinygrad/llm/cli.py test/external/external_test_llm_xgrammar.py
git commit -m "test: add qwen3.5 2b xgrammar smoke tests"
```

### Task 6: Run full verification and portability checks

**Objective:** Ensure the change is correct, does not regress existing behavior, and remains backend-agnostic.

**Files:**
- Modify only if failures require fixes in files above

**Step 1: Run focused unit/integration suites**

```bash
python3.11 -m pytest test/unit/test_llm_server.py test/null/test_llm_server.py -q
```

Expected: PASS.

**Step 2: Run external xgrammar smoke tests**

```bash
python3.11 -m pytest test/external/external_test_llm_xgrammar.py -q -s
```

Expected: PASS or clean skip when runtime/model prerequisites are absent.

**Step 3: Run non-LLM regression suite that touches changed behavior**

```bash
python3.11 -m pytest test/null/test_llm_tokenizer.py -q
```

Expected: PASS.

**Step 4: Manual server verification on macOS path**

Run from `/Users/surya/Documents/Projects/tinygrad` with Qwen3.5 2B and verify:
- structured JSON response
- single tool call
- parallel tool call

**Step 5: Commit**

```bash
git add -A
git commit -m "test: verify xgrammar llm support"
```

---

## Review Checklist

- [ ] `Transformer` exposes logits before sampling
- [ ] xgrammar integration is optional and lazily imported
- [ ] structured outputs work through `response_format`
- [ ] tool calling works for auto/required/forced choice
- [ ] parallel tool calling can be toggled on/off
- [ ] streaming and non-streaming responses match OpenAI expectations
- [ ] assistant/tool history is serialized for follow-up turns
- [ ] tests cover unit, integration, and real Qwen3.5 2B smoke cases
- [ ] implementation does not assume Apple-only APIs and relies only on tinygrad backends
