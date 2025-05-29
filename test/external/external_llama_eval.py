from lm_eval import simple_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from pathlib import Path
import json, argparse

from examples.llama3 import build_transformer, Tokenizer, MODEL_PARAMS
from tinygrad.tensor import Tensor
from tinygrad.helpers import tqdm
from tinygrad import Device

class LLaMaAdaptor(LM):
  def __init__(
    self,
    model_size: str,
    checkpoint_path: Path,
    is_chat_model: bool,
    max_length: int,
    quantize: bool,
  ):
    super().__init__()
    self.max_length = max_length
    self.is_chat_model = is_chat_model
    self.tokenizer = Tokenizer(str((checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent) / "tokenizer.model"))
    self.model = build_transformer(checkpoint_path, model_size, quantize)
  def encode_role(self, role: str):
    return [self.tokenizer.special_tokens["<|start_header_id|>"]] + self.tokenizer.encode(role) + \
      [self.tokenizer.special_tokens["<|end_header_id|>"]] + self.tokenizer.encode("\n\n")
  def encode_message(self, role: str, content: str):
    return self.encode_role(role) + self.tokenizer.encode(content.strip()) + [self.tokenizer.special_tokens["<|eot_id|>"]]
  def generate_until(self, requests: list[Instance]) -> list[str]:
    continuations = []
    for request in tqdm(requests):
      prompt, args = request.args
      temperature = args.get("temperature", 0.0)
      max_length = args.get("max_length", self.max_length)
      until = [self.tokenizer.encode(tok) for tok in args.get("until", [])]
      if self.is_chat_model:
        toks = [self.tokenizer.bos_id] + self.encode_message("system", "You are a helpful assistant") + \
          self.encode_message("user", prompt) + self.encode_role("assistant")
      else:
        toks = [self.tokenizer.bos_id] + self.tokenizer.encode(prompt)
      start_pos = prompt_len = len(toks)
      for i in range(max_length):
        next_tok = self.model(Tensor([toks]), start_pos, temperature, top_p=0.0).item()
        if next_tok in self.tokenizer.stop_tokens or next_tok in until: break
        toks.append(next_tok)
        start_pos = len(toks)
      continuations.append(self.tokenizer.decode(toks[prompt_len:]))
    return continuations
  def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]: raise NotImplementedError() # not needed for gsm8k
  def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]: raise NotImplementedError()

if __name__ == '__main__':
  print(f"using {Device.DEFAULT} backend")

  parser = argparse.ArgumentParser(description='Run LLaMA evals in tinygrad', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--size', type=str, default="8B", help=f"Size of model to use [{', '.join(list(MODEL_PARAMS.keys()))}]")
  parser.add_argument('--chat', action='store_true', help="Use chat model")
  parser.add_argument('--ctx', type=int, default=128000, help="Max context length")
  parser.add_argument('--quantize', type=str, default=None, help="Quantize the weights to int8 or int4 in memory")
  parser.add_argument('--eval', type=str, default="gsm8k_cot_llama", help="Run evaluation")
  parser.add_argument('--limit', type=int, default=None, help="Limit tests in eval")
  parser.add_argument('--num_fewshot', type=int, default=None, help="Limit tries(starts with 0)")
  parser.add_argument('--model', type=str, default="./weights/LLaMa/", help="Location of the weights")
  args = parser.parse_args()

  # run eval and exit
  adaptor = LLaMaAdaptor(model_size=args.size, quantize=args.quantize,
                         checkpoint_path=Path(args.model), is_chat_model=args.chat, max_length=args.ctx)
  results = simple_evaluate(
    model=adaptor,
    tasks=args.eval.split(","),
    num_fewshot=args.num_fewshot,
    task_manager=None,
    limit=args.limit
  )
  print(json.dumps(results, indent=2))
