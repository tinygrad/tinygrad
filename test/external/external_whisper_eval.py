import sys, math, string, argparse, difflib
from examples.whisper import make_initial_prompt, transcribe_wav, WHISPER_MODELS, load_whisper_model
from extra.datasets.librispeech import ci, BASEDIR
from examples.mlperf.metrics import word_error_rate
import numpy as np

WER = {}

def output_wer(wer: dict):
  for (k,v) in wer.items():
    print(f"{k}: {(np.average(v)*100):.2f}%")
    print(f"{k}: {np.count_nonzero(v)} out of {len(v)} samples have errors, {(np.count_nonzero(v)/len(v)*100):.2f}%")

def eval_whisper(model, model_name, start, end, verbose=2):
  diff = difflib.Differ()
  for c in ci[start:end]:
    fn = BASEDIR / c["files"][0]["fname"]
    predicted = "".join(transcribe_wav(fn, model, make_initial_prompt(model, "en"))).translate(str.maketrans("", "", string.punctuation)).lower()
    transcript = c["transcript"].translate(str.maketrans("", "", string.punctuation))
    current_wer = word_error_rate([predicted], [transcript])[0]
    WER[model_name] = np.append(WER[model_name], current_wer)
    if (verbose > 0 and predicted != transcript) or (verbose > 1):
      print("-" * 128, f"{fn.stem}\n", sep="\n")
      sys.stdout.writelines(list(diff.compare([predicted + "\n"], [transcript + "\n"])))
      print(f"\nword error rate: {current_wer:.4f}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Evaluate whisper on librispeech', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--models', type=str, default=None, nargs="+", help="Which model to evaluate, if not specified, will use eval all available models")
  parser.add_argument('--verbose', type=int, default=2, help="Verbosity level, 0: only print final WER, 1: print WER only for failed samples, 2: print WER for all samples")
  parser.add_argument('--single', action='store_true', help="Run all models on single sample, to check whether they are working")
  parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to run on")
  parser.add_argument("--step-size", type=int, default=None, help="Each step it runs all models on all samples in a step, ")
  args = parser.parse_args()
  models = WHISPER_MODELS if args.models is None else {x:WHISPER_MODELS[x] for x in args.models if x in WHISPER_MODELS}
  # large-v2 appears twice in the list, usually it's not the problem, but here we load a bunch of models, so it's better to remove it
  if "large" in models:
    models["large-v2"] = models["large"]
    del models["large"]
  num_samples = len(ci) if args.num_samples is None else min(args.num_samples, len(ci))
  step_size = num_samples if args.step_size is None else min(args.step_size, num_samples)
  WER = {j:np.array([]) for j in models}
  if args.single:
    num_samples = 1
    step_size = 1
  for i in range(0, num_samples, step_size):
    for j in models:
      print(f"evaluating {j} on {step_size} sample(s)")
      model = load_whisper_model(j)
      eval_whisper(model, j, i, max(i+step_size, num_samples), verbose=args.verbose)
      if args.verbose > 0: 
        print("-"*128)
      del model
  print("Results of a run:")
  output_wer(WER)