#!/usr/bin/env python3
"""Evaluate the tinygrad LLM server using lm-evaluation-harness.

Requires:
  pip install "lm-eval[api]"

Usage:
  # start the server first
  python -m tinygrad.apps.llm --serve

  # run GSM8K (default)
  python extra/eval_llm.py

  # run with options
  python extra/eval_llm.py --task gsm8k --limit 100 --port 11434

  # run multiple tasks
  python extra/eval_llm.py --task gsm8k,ifeval
"""
import argparse, subprocess, sys

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Evaluate tinygrad LLM server with lm-evaluation-harness")
  parser.add_argument("--task", "-t", type=str, default="gsm8k", help="Task(s) to evaluate, comma-separated (default: gsm8k)")
  parser.add_argument("--port", "-p", type=int, default=11434, help="Server port (default: 11434)")
  parser.add_argument("--limit", "-L", type=int, default=None, help="Limit number of examples (for testing)")
  parser.add_argument("--num_fewshot", "-f", type=int, default=0, help="Number of few-shot examples (default: 0)")
  parser.add_argument("--model", "-m", type=str, default="llama3.2:1b", help="Model name to pass to the API (default: llama3.2:1b)")
  args = parser.parse_args()

  cmd = [
    sys.executable, "-m", "lm_eval",
    "--model", "local-chat-completions",
    "--model_args", f"model={args.model},base_url=http://127.0.0.1:{args.port}/v1/chat/completions,tokenized_requests=False",
    "--tasks", args.task,
    "--num_fewshot", str(args.num_fewshot),
    "--apply_chat_template",
  ]
  if args.limit is not None: cmd += ["--limit", str(args.limit)]

  env = {"OPENAI_API_KEY": "tinygrad"}
  sys.exit(subprocess.call(cmd, env={**__import__("os").environ, **env}))
