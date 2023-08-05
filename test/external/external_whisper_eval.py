import sys, math, string, argparse, difflib
from examples.whisper import transcribe_wav, WHISPER_MODELS, load_whisper_model
from extra.datasets.librispeech import ci, BASEDIR
from examples.mlperf.metrics import word_error_rate

WER = {}

def output_wer(wer: dict):
  for (k,v) in wer:
    print(f"{k}: {sum(v)/len(v):.4f}%")

def eval_whisper(model, verbose=2):
  diff = difflib.Differ()
  for c in ci:
    fn = BASEDIR / c["files"][0]["fname"]
    print("-" * 128, f"{fn.stem}\n", sep="\n")
    predicted = "".join(transcribe_wav(fn, model, ["<|startoftranscript|>", "<|en|>", "<|transcribe|>"])).translate(str.maketrans("", "", string.punctuation)).lower()
    transcript = c["transcript"].translate(str.maketrans("", "", string.punctuation))
    if (verbose > 0 and predicted != transcript) or (verbose > 1):
      sys.stdout.writelines(list(diff.compare([predicted + "\n"], [transcript + "\n"])))
      print(f"\nword error rate: {word_error_rate([predicted], [transcript])[0]:.4f}")
    else:
      print("passed")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Evaluate whisper on librispeech', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--models', type=str, default=None, nargs="+", help="Which model to evaluate, if not specified, will use eval all available models")
  parser.add_argument('--verbose', type=int, default=2, help="Verbosity level, 0: only print final wer, 1: print wer only for failed samples, 2: print wer for all samples")
  parser.add_argument('--single', action='store_true', help="Run all models on single sample, to check whether they are working")
  parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to run on")
  parser.add_argument("--step-size", type=int, default=None, help="Each step it runs all models on all samples in a step, ")
  args = parser.parse_args()
  models = WHISPER_MODELS if args.models is None else args.models
  num_samples = len(ci) if args.num_samples is None else min(args.num_samples, len(ci))
  step_size = num_samples if args.step_size is None else min(args.step_size, num_samples)
  if args.single:
    num_samples = 1
    step_size = 1
  try:
    for i in range(0, num_samples, step_size):
      for j in models:
        model = load_whisper_model(j)
        print(f"evaluating {j} on {step_size} samples")
        eval_whisper(model, )
  except KeyboardInterrupt:
    output_wer(WER)