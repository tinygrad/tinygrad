import argparse
from tinygrad.llm.kimi import convert_kimi

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Convert official Kimi-Linear-48B-A3B BF16 weights to tinygrad MXFP4/BF16")
  parser.add_argument("source", help="downloaded moonshotai/Kimi-Linear-48B-A3B-Instruct directory")
  parser.add_argument("output", help="output directory")
  args = parser.parse_args()
  convert_kimi(args.source, args.output)
