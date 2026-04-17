from typing import Any

from tinygrad.helpers import getenv

LLAMA2_70B_LORA_PROMPT_PREFIX = "### Summarize the following text:\n "
LLAMA2_70B_LORA_PROMPT_SUFFIX = "\n ### Summary:\n"


def llama_benchmark_config(model_name:str, *, small:bool=False) -> dict[str, Any]:
  if model_name == "llama3":
    from examples.llama3 import MODEL_PARAMS
    size = getenv("LLAMA3_SIZE", "8B")
    model_params = dict(MODEL_PARAMS[size]["args"])
    if not small: model_params["vocab_size"] = 32000
    benchmark_name = "llama3.1_405b" if size == "405B" else "llama3.1_8b"
    return {
      "benchmark_steps": (lambda gbs, max_steps: 200_000 // gbs if size == "8B" else max_steps),
      "checkpoint_prefix": "llama3",
      "model_params": model_params,
      "real_vocab_size": model_params["vocab_size"],
      "result_prefix": "llama31",
      "submission_benchmark": benchmark_name,
      "wandb_project": "MLPerf-LLaMA3",
    }
  if model_name == "llama2_70b_lora":
    model_params = {
      "dim": 8192, "hidden_dim": 28672, "n_heads": 64, "n_kv_heads": 8,
      "n_layers": 80, "norm_eps": 1e-5, "rope_theta": 10000, "vocab_size": 32000,
    }
    return {
      "benchmark_steps": (lambda _gbs, max_steps: max_steps),
      "checkpoint_prefix": "llama2_70b_lora",
      "lora_alpha": 32,
      "lora_dropout": 0.1,
      "lora_rank": 16,
      "model_params": model_params,
      "real_vocab_size": model_params["vocab_size"],
      "result_prefix": "llama2_70b_lora",
      "submission_benchmark": "llama2_70b_lora",
      "wandb_project": "MLPerf-LLaMA2-70B-LoRA",
    }
  raise ValueError(f"unsupported LLaMA benchmark {model_name}")


def llama_model_state_dict(state_dict:dict[str, Any]) -> dict[str, Any]:
  return {k.removeprefix("model."): v for k, v in state_dict.items() if k.startswith("model.")} or state_dict


def llama2_70b_lora_prompt(source:str) -> str:
  return f"{LLAMA2_70B_LORA_PROMPT_PREFIX}{source}{LLAMA2_70B_LORA_PROMPT_SUFFIX}"


def llama2_70b_lora_encode_sample(tokenizer, source:str, target:str) -> tuple[list[int], list[int]]:
  prompt_tokens = [tokenizer.bos_id(), *tokenizer.encode(llama2_70b_lora_prompt(source))]
  target_tokens = tokenizer.encode(target)
  input_ids = [*prompt_tokens, *target_tokens, tokenizer.eos_id()]
  labels = input_ids.copy()
  labels[:len(prompt_tokens)] = [-1] * len(prompt_tokens)
  labels[-1] = -1
  return input_ids, labels
