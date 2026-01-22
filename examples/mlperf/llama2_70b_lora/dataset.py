# GovReport dataset for Llama2 summarization task
import json, random
from pathlib import Path
from tinygrad import Tensor

IGNORE_IDX = -100
PROMPT = "Summarize the following government document:\n\n{input}\n\nSummary:"

def load_data(path, split="train"):
  p = Path(path)
  f = p / f"{split}.json"
  if not f.exists():
    p.mkdir(parents=True, exist_ok=True)
    dummy = [{"input": "Sample government policy report. "*50, "output": "Policy implementation summary.", "id": f"d{i}"} for i in range(10)]
    for s in ["train", "validation", "test"]: json.dump(dummy, open(p/f"{s}.json",'w'), indent=2)
    print(f"created dummy data: {len(dummy)} examples/split")
  data = json.load(open(f))
  print(f"loaded {len(data)} {split} examples")
  return data

def tokenize_ex(ex, tok, maxlen):
  inp = [tok.bos_token_id] + tok.encode(PROMPT.format(input=ex['input']))
  tgt = tok.encode(ex['output'])
  toks = inp + tgt + [tok.eos_token_id]
  labels = [IGNORE_IDX]*len(inp) + tgt + [tok.eos_token_id]
  if len(toks) > maxlen: toks, labels = toks[:maxlen], labels[:maxlen]
  alen = len(toks)
  attn = [1]*alen + [0]*(maxlen-alen)
  toks += [tok.pad_token_id]*(maxlen-len(toks))
  labels += [IGNORE_IDX]*(maxlen-len(labels))
  return {'input_ids': toks, 'attention_mask': attn, 'labels': labels}

def batch_iter(data, tok, bs, maxlen, shuffle=True):
  idxs = list(range(len(data)))
  if shuffle: random.shuffle(idxs)
  for i in range(0, len(data), bs):
    bidx = idxs[i:i+bs]
    if len(bidx) < bs: continue
    batch = [tokenize_ex(data[j], tok, maxlen) for j in bidx]
    yield {
      'input_ids': Tensor([x['input_ids'] for x in batch], dtype='int32'),
      'attention_mask': Tensor([x['attention_mask'] for x in batch], dtype='int32'),
      'labels': Tensor([x['labels'] for x in batch], dtype='int32')
    }

def get_tokenizer(mp=None):
  from tinygrad.helpers import fetch
  import sentencepiece as spm
  #try to load from model dir first
  if mp:
    mdir = Path(mp).parent if Path(mp).is_file() else Path(mp)
    tok_model = mdir / "tokenizer.model"
    if tok_model.exists():
      sp = spm.SentencePieceProcessor(model_file=str(tok_model))
      class SPTok:
        def __init__(self, sp): self.sp, self.pad_token_id, self.bos_token_id, self.eos_token_id = sp, sp.pad_id(), sp.bos_id(), sp.eos_id()
        def encode(self, t): return self.sp.encode(t, out_type=int)
        def decode(self, ids): return self.sp.decode(ids)
      return SPTok(sp)
  #fallback: use TinyLlama tokenizer (no auth required) - compatible with Llama-2
  url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.model"
  tok_path = fetch(url, "llama_tokenizer.model")
  sp = spm.SentencePieceProcessor(model_file=str(tok_path))
  class SPTok:
    def __init__(self, sp): self.sp, self.vocab_size, self.pad_token_id, self.bos_token_id, self.eos_token_id = sp, sp.vocab_size(), 0, 1, 2
    def encode(self, t): return self.sp.encode(t, out_type=int)
    def decode(self, ids): return self.sp.decode(ids)
  return SPTok(sp)
