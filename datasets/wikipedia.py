import sys
import pickle
from pathlib import Path
from transformers import BertTokenizer
import numpy as np
from tqdm import tqdm
import random
import unicodedata

BASEDIR = Path(__file__).parent.parent / "datasets/wikipedia"

def _is_punctuation(char):
  if (cp := ord(char)) in range(33, 48) or cp in range(58, 65) or cp in range(91, 97) or cp in range(123, 127):
    return True
  return unicodedata.category(char).startswith("P")

def _is_whitespace(char):
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  return unicodedata.category(char) == "Zs"

def _is_control(char):
  if char == "\t" or char == "\n" or char == "\r":
    return False
  return unicodedata.category(char).startswith("C")

def _run_split_on_punc(text):
  if text in ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"):
    return [text]
  start_new_word = True
  output = []
  for i in range(len(text)):
    if _is_punctuation(char := text[i]):
      output.append([char])
      start_new_word = True
    else:
      if start_new_word:
        output.append([])
      start_new_word = False
      output[-1].append(char)
  return ["".join(x) for x in output]

def _run_strip_accents(text):
  output = []
  for char in unicodedata.normalize("NFD", text):
    if unicodedata.category(char) != "Mn":
      output.append(char)
  return "".join(output)

def _clean_text(text):
  output = []
  for char in text:
    if not ((cp := ord(char)) == 0 or cp == 0xfffd or _is_control(char)):
      output.append(" " if _is_whitespace(char) else char)
  return "".join(output)

def _is_chinese_char(cp):
  return cp in range(0x4E00, 0x9FFF+1) or cp in range(0x3400, 0x4DBF+1) or cp in range(0x20000, 0x2A6DF+1) or cp in range(0x2A700, 0x2B73F+1) or cp in range(0x2B740, 0x2B81F+1) or cp in range(0x2B820, 0x2CEAF+1) or cp in range(0xF900, 0xFAFF+1) or cp in range(0x2F800, 0x2FA1F+1)

def _tokenize_chinese_chars(text):
  output = []
  for char in text:
    cp = ord(char)
    if _is_chinese_char(cp):
      output.append(" ")
      output.append(char)
      output.append(" ")
    else:
      output.append(char)
  return "".join(output)

def _wordpiece_tokenize(text, vocab):
  text = text.decode("utf-8", "ignore") if isinstance(text, bytes) else text
  output_tokens = []
  for token in text.strip().split():
    chars = list(token)
    if len(chars) > 200:
      output_tokens.append("[UNK]")
      continue

    is_bad = False
    start = 0
    sub_tokens = []
    while start < len(chars):
      end = len(chars)
      cur_substr = None
      while start < end:
        substr = "".join(chars[start:end])
        if start > 0:
          substr = "##" + substr
        if substr in vocab:
          cur_substr = substr
          break
        end -= 1
      if cur_substr is None:
        is_bad = True
        break
      sub_tokens.append(cur_substr)
      start = end

    if is_bad:
      output_tokens.append("[UNK]")
    else:
      output_tokens.extend(sub_tokens)
  return output_tokens

class Tokenizer:
  def __init__(self, vocab_file):
    self.vocab = {}
    with open(vocab_file) as f:
      for line in f:
        line = line.decode("utf-8", "ignore") if isinstance(line, bytes) else line
        if (token := line.strip()) and token not in self.vocab:
          self.vocab[token] = len(self.vocab)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}

  def tokenize(self, text):
    text = _clean_text(text.decode("utf-8", "ignore") if isinstance(text, bytes) else text)
    text = _tokenize_chinese_chars(text)

    # BasicTokenizer
    split_tokens = []
    for token in text.strip().split():
      split_tokens.extend(_run_split_on_punc(_run_strip_accents(token.lower())))
    split_tokens = " ".join(split_tokens).strip().split()

    # WordpieceTokenizer
    tokens = []
    for token in split_tokens:
      tokens.extend(_wordpiece_tokenize(token, self.vocab))

    return tokens

  def convert_tokens_to_ids(self, tokens):
    return [self.vocab[token] for token in tokens]

  def convert_ids_to_tokens(self, ids):
    return [self.inv_vocab[id] for id in ids]

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

def create_masked_lm_predictions(tokens, rng, vocab_words):
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
        masked_token = vocab_words[rng.randint(0, len(tokenizer.vocab) - 1)]

    output_tokens[index] = masked_token
    masked_lms.append((index, tokens[index]))
  masked_lms = sorted(masked_lms, key=lambda x: x[0])

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p[0])
    masked_lm_labels.append(p[1])

  return output_tokens, masked_lm_positions, masked_lm_labels

def get_documents(rng, tokenizer, fn):
  documents = [[]]
  with open(BASEDIR / "raw" / fn) as f:
    for line in tqdm(f.readlines()):
      if not (line := line.decode("utf-8", "ignore") if isinstance(line, bytes) else line): break
      if not (line := line.strip()): documents.append([])
      if (tokens := tokenizer.tokenize(line)): documents[-1].append(tokens)
  documents = [x for x in documents if x]
  rng.shuffle(documents)
  return documents

def create_instances_from_document(rng, tokenizer, doc, di, documents):
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

        tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(tokens, rng, list(tokenizer.vocab.keys()))
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
  return instances

def get_instances(rng, tokenizer, documents):
  instances = []
  for i in range(1):
    for di, doc in tqdm(enumerate(documents), desc=f"dupe {i}", total=len(documents)):
      instances.extend(create_instances_from_document(rng, tokenizer, doc, di, documents))
  rng.shuffle(instances)
  return instances

def process_iterate(tokenizer, val=True):
  rng = random.Random(12345)

  if val:
    documents = get_documents(rng, tokenizer, "eval.txt")
    instances = get_instances(rng, tokenizer, documents)

    print(f"there are {len(instances)} samples in the dataset")
    print(f"picking 10000 samples")

    pick_ratio = len(instances) // 10000
    for i in range(10000):
      instance = instances[i * pick_ratio]
      input_ids = tokenizer.convert_tokens_to_ids(instance["tokens"])
      input_mask = [1] * len(input_ids)
      segment_ids = instance["segment_ids"]

      assert len(input_ids) <= 512
      while len(input_ids) < 512:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
      assert len(input_ids) == 512
      assert len(input_mask) == 512
      assert len(segment_ids) == 512

      masked_lm_positions = instance["masked_lm_positions"]
      masked_lm_ids = tokenizer.convert_tokens_to_ids(instance["masked_lm_labels"])

      while len(masked_lm_positions) < 76:
        masked_lm_positions.append(0)
        masked_lm_ids.append(0)

      next_sentence_label = 1 if instance["is_random_next"] else 0

      features = {
        "input_ids": np.expand_dims(np.array(input_ids, dtype=np.float32), 0),
        "input_mask": np.expand_dims(np.array(input_mask, dtype=np.float32), 0),
        "segment_ids": np.expand_dims(np.array(segment_ids, dtype=np.float32), 0),
        "masked_lm_positions": np.expand_dims(np.array(masked_lm_positions, dtype=np.float32), 0),
        "masked_lm_ids": np.expand_dims(np.array(masked_lm_ids, dtype=np.float32), 0),
        "next_sentence_labels": np.expand_dims(np.array([next_sentence_label], dtype=np.float32), 0),
      }

      yield features, instances[i]

def get_val_files():
  return sorted(list((BASEDIR / "eval").glob("*.pkl")))

def iterate(bs=1, start=0, val=True):
  if val:
    # scan directory for files
    files = get_val_files()
    for i in range(start, len(files), bs):
      input_ids = []
      input_mask = []
      segment_ids = []
      masked_lm_positions = []
      masked_lm_ids = []
      next_sentence_labels = []
      instances = []
      for j in range(bs):
        with open(files[i + j], "rb") as f:
          features, instance = pickle.load(f)
          input_ids.append(features["input_ids"])
          input_mask.append(features["input_mask"])
          segment_ids.append(features["segment_ids"])
          masked_lm_positions.append(features["masked_lm_positions"])
          masked_lm_ids.append(features["masked_lm_ids"])
          next_sentence_labels.append(features["next_sentence_labels"])
          instances.append(instance)

      yield {
        "input_ids": np.concatenate(input_ids, axis=0),
        "input_mask": np.concatenate(input_mask, axis=0),
        "segment_ids": np.concatenate(segment_ids, axis=0),
        "masked_lm_positions": np.concatenate(masked_lm_positions, axis=0),
        "masked_lm_ids": np.concatenate(masked_lm_ids, axis=0),
        "next_sentence_labels": np.concatenate(next_sentence_labels, axis=0),
      }, instances

if __name__ == "__main__":
  # tokenizer = BertTokenizer(str(Path(__file__).parent.parent / "weights/bert_vocab.txt"), do_lower_case=True)
  tokenizer = Tokenizer(Path(__file__).parent.parent / "weights/bert_vocab.txt")

  if len(sys.argv) <= 1:
    X, Y = next(iterate())
    print(X["input_ids"])
    print(tokenizer.convert_ids_to_tokens(X["input_ids"][0]))
    print(X["masked_lm_ids"])
    print(tokenizer.convert_ids_to_tokens(X["masked_lm_ids"][0]))

    # fill in the blanks
    for i in range(76):
      X["input_ids"][0][int(X["masked_lm_positions"][0][i])] = X["masked_lm_ids"][0][i]
    print(" ".join(tokenizer.convert_ids_to_tokens(X["input_ids"][0])))
  else:
    if sys.argv[1] == "pre-eval":
      for i, (X, Y) in tqdm(enumerate(process_iterate(tokenizer)), total=10000):
        with open(BASEDIR / f"eval/{i}.pkl", "wb") as f:
          pickle.dump((X, Y), f)
