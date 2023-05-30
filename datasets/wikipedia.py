import json
import os
from pathlib import Path
from transformers import BertTokenizer
import numpy as np
from extra.utils import download_file
import random

BASEDIR = Path(__file__).parent.parent / "datasets/wikipedia"

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()

def create_masked_lm_predictions(tokens, rng, tokenizer):
  cand_indices = []
  for i, token in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    cand_indices.append(i)

  rng.shuffle(cand_indices)
  output_tokens = list(tokens)
  num_to_predict = min(76, max(1, int(round(len(tokens) * 0.15))))

  masked_lms = []
  covered_indices = set()
  for index in cand_indices:
    if len(masked_lms) >= num_to_predict:
      break
    if index in covered_indices:
      continue
    covered_indices.add(index)

    masked_token = None
    if rng.random() < 0.8:
      masked_token = "[MASK]"
    else:
      if rng.random() < 0.5:
        masked_token = tokens[index]
      else:
        masked_token = tokenizer.convert_tokens_to_ids([tokens[index]])[0]

    output_tokens[index] = masked_token
    masked_lms.append((index, tokens[index]))
  masked_lms = sorted(masked_lms, key=lambda x: x[0])

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p[0])
    masked_lm_labels.append(p[1])

  return output_tokens, masked_lm_positions, masked_lm_labels

def iterate(tokenizer, start=0, val=True):
  rng = random.Random(12345)

  if val:
    documents = [[]]
    with open(BASEDIR / "eval.txt") as f:
      for line in f:
        if not line:
          break
        line = line.strip()
        if not line:
          documents.append([])
          if len(documents) > 2:
            break
        tokens = tokenizer.tokenize(line)
        if tokens: documents[-1].append(tokens)
    documents = [x for x in documents if x]
    rng.shuffle(documents)

    all_instances = []
    for di, doc in enumerate(documents):
      print(f"processing document {di} of {len(documents)}")
      max_num_tokens = 512 - 3

      target_seq_length = max_num_tokens
      if rng.random() < 0.1:
        target_seq_length = rng.randint(2, max_num_tokens)

      instances = []
      current_chunk = []
      current_length = 0
      i = 0
      while i < len(doc):
        segment = doc[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(doc) - 1 or current_length >= target_seq_length:
          if current_chunk:
            a_end = 1
            if len(current_chunk) >= 2:
              a_end = rng.randint(1, len(current_chunk) - 1)

            tokens_a = []
            for j in range(a_end):
              tokens_a.extend(current_chunk[j])

            tokens_b = [] 
            is_random_next = False
            if len(current_chunk) == 1 or rng.random() < 0.5:
              is_random_next = True
              target_b_length = target_seq_length - len(tokens_a)

              for _ in range(10):
                random_document_index = rng.randint(0, len(documents) - 1)
                if random_document_index != di:
                  break

              random_document = documents[random_document_index]
              random_start = rng.randint(0, len(random_document) - 1)
              for j in range(random_start, len(random_document)):
                tokens_b.extend(random_document[j])
                if len(tokens_b) >= target_b_length:
                  break

              num_unused_segments = len(current_chunk) - a_end
              i -= num_unused_segments
            else:
              is_random_next = False
              for j in range(a_end, len(current_chunk)):
                tokens_b.extend(current_chunk[j])
            truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in tokens_a:
              tokens.append(token)
              segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)
            for token in tokens_b:
              tokens.append(token)
              segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(tokens, rng, tokenizer)
            instances.append({
              "tokens": tokens,
              "segment_ids": segment_ids,
              "masked_lm_positions": masked_lm_positions,
              "masked_lm_labels": masked_lm_labels,
              "is_random_next": is_random_next
            })
          current_chunk = []
          current_length = 0
        i += 1
      all_instances.extend(instances)

    print(f"there are {len(all_instances)} samples in the dataset")
    for i in range(start, len(all_instances)):
      input_ids = tokenizer.convert_tokens_to_ids(all_instances[i]["tokens"])
      input_mask = [1] * len(input_ids)
      segment_ids = all_instances[i]["segment_ids"]

      assert len(input_ids) <= 512
      while len(input_ids) < 512:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
      assert len(input_ids) == 512
      assert len(input_mask) == 512
      assert len(segment_ids) == 512

      masked_lm_positions = all_instances[i]["masked_lm_positions"]
      masked_lm_ids = tokenizer.convert_tokens_to_ids(all_instances[i]["masked_lm_labels"])
      masked_lm_weights = [1.0] * len(masked_lm_ids)

      while len(masked_lm_positions) < 76:
        masked_lm_positions.append(0)
        masked_lm_ids.append(0)
        masked_lm_weights.append(0.0)

      next_sentence_label = 1 if all_instances[i]["is_random_next"] else 0

      features = {
        "input_ids": np.expand_dims(np.array(input_ids), 0).astype(np.float32),
        "input_mask": np.expand_dims(np.array(input_mask), 0).astype(np.float32),
        "segment_ids": np.expand_dims(np.array(segment_ids), 0).astype(np.float32),
        "masked_lm_positions": masked_lm_positions,
        "masked_lm_ids": masked_lm_ids,
        "masked_lm_weights": masked_lm_weights,
        "next_sentence_labels": [next_sentence_label]
      }

      yield features, all_instances[i]

if __name__ == "__main__":
  tokenizer = BertTokenizer(str(Path(__file__).parent.parent / "weights/bert_vocab.txt"))

  X, Y = next(iterate(tokenizer))
  print(" ".join(map(str, Y["tokens"])))
