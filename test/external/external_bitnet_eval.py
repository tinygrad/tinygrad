from lm_eval import evaluator, tasks
import torch, json, argparse

from examples.bitnet import build_transformer
from tinygrad.tensor import Tensor
from tinygrad import Device
from pathlib import Path

class BitNetAdaptor():
  def __init__(
    self,
    model_size="7B",
    model_gen=1,
    device="",
    quantize=False,
    batch_size=1,
    max_batch_size=1,
    do_sample=False,
    temperature=1.0,
    checkpoint_path="",
    tokenizer_path="",
  ):
    if batch_size is None:
      batch_size = 1
    self.do_sample = do_sample
    self.temperature = temperature
    self._device = device

    assert isinstance(model_gen, int)
    assert isinstance(model_size, str)
    assert isinstance(batch_size, int)
    assert isinstance(checkpoint_path, str)
    assert isinstance(tokenizer_path, str)

    self.bitnet = build_transformer(model_path=Path(checkpoint_path), model_size=model_size, quantize=quantize, device=self._device)

  @classmethod
  def create_from_arg_string(cls, arg_string, additional_config=None):
    kwargs = {el.split("=")[0]: el.split("=")[1] for el in arg_string.split(",")}
    return cls(**kwargs, **additional_config)

  @property
  def eot_token_id(self):
    # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
    return self.llama.tokenizer.eos_id()

  @property
  def max_length(self):
    return 1024

  @property
  def max_gen_toks(self):
    return 256

  @property
  def batch_size(self):
    return 1

  @property
  def device(self):
    return self._device

  def tok_encode(self, string: str):
    return [self.llama.tokenizer.bos_id()] + self.llama.tokenizer.encode(string)

  def tok_decode(self, tokens):
    return self.llama.tokenizer.decode(tokens)

  def _model_call(self, inps):
    Tensor.no_grad = True
    return torch.Tensor(self.llama.model(Tensor(inps.numpy()), 0).numpy())

  def greedy_until(self, requests):
    continuations = []
    for request in requests:
      prompt, until = request[0], request[1]['until']
      output = self.llama.greedy_until(prompt, until, max_length=128, temperature=0.0)
      continuations.append(output[len(prompt):])
    return continuations

  def _model_generate(self, context, max_length, eos_token_id):
    raise NotImplementedError()

if __name__ == '__main__':
  print(f"using {Device.DEFAULT} backend")

  parser = argparse.ArgumentParser(description='Run Bitnet evals in tinygrad', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--size', type=str, default="2B", help="Size of model to use [2B]")
  parser.add_argument('--gen', type=int, default="1", help="Generation of the model to use [1]")
  parser.add_argument('--quantize', action='store_true', help="Quantize the weights to int8 in memory")
  parser.add_argument('--eval', type=str, default="arc_easy", help="Run in evaluation mode")
  parser.add_argument('--limit', type=int, default=None, help="Limit tests in eval")
  parser.add_argument('--weights', type=str, default="./weights/Bitnet/", help="Location of the weights")
  parser.add_argument('--tokenizer', type=str, default="./weights/Bitnet/tokenizer.json", help="Location of the tokenizer")
  args = parser.parse_args()

  # run eval and exit
  adaptor = BitNetAdaptor(model_gen=args.gen, model_size=args.size, quantize=args.quantize,
                         checkpoint_path=args.weights, tokenizer_path=args.tokenizer, device="cpu")
  results = evaluator.evaluate(adaptor, tasks.get_task_dict(args.eval.split(",")), False, 0, args.limit)
  print(json.dumps(results, indent=2))
