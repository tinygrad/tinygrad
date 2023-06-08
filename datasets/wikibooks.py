# based on nanoGPT's prepare of openwebtext
# https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py

import os
import re
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer
from datasets import load_dataset, concatenate_datasets  # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 13

# prepare bookcorpus
bookcorpus = load_dataset("bookcorpus", split='train', num_proc=num_proc)

# prepare wiki
wiki = load_dataset("wikipedia", "20220301.en", split='train', num_proc=num_proc)
wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])

def normalize_whitespaces(example):
  text = example['text']
  text = re.sub('\s+', ' ', text)
  return {'text': text}

wiki = wiki.map(normalize_whitespaces, num_proc=num_proc)

# merge bookcorpus and wiki
dataset = concatenate_datasets([bookcorpus, wiki])

# owt by default only contains the 'train' split, so create a test split
split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')  # rename the test split to val

# we now want to tokenize the dataset.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def process(example):
  ids = tokenizer(example['text'])['input_ids']
  out = {'ids': ids, 'len': len(ids)}
  return out

# tokenize the dataset
tokenized = split_dataset.map(
  process,
  remove_columns=['text'],
  desc="tokenizing the splits",
  num_proc=num_proc,
)

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
  arr_len = np.sum(dset['len'])
  filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
  dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
  arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

  print(f"writing {filename}...")
  idx = 0
  for example in tqdm(dset):
    arr[idx:idx + example['len']] = example['ids']
    idx += example['len']
  arr.flush()

# train.bin is ~10.3GB, val.bin ~5.5MB
# train has ~11B tokens (11,037,086,592) 
# val has ~6M tokens (5,723,362)

# to read the bin files later, e.g. with numpy:
# m = np.memmap('train.bin', dtype=np.uint16, mode='r')