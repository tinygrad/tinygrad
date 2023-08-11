#!/usr/bin/env python
import unittest
from pathlib import Path
import numpy as np
from tinygrad.tensor import Tensor
from extra.datasets.squad import iterate
import torch

def get_question_samp(bsz, seq_len, vocab_size, seed):
  np.random.seed(seed)
  in_ids= np.random.randint(vocab_size, size=(bsz, seq_len))
  mask = np.random.choice([True, False], size=(bsz, seq_len))
  seg_ids = np.random.randint(1, size=(bsz, seq_len))
  return in_ids, mask, seg_ids

class TestBert(unittest.TestCase):
  def test_questions(self):
    from models.bert import BertForQuestionAnswering
    from transformers import BertForQuestionAnswering as TorchBertForQuestionAnswering
    from transformers import BertConfig

    config = {
      'vocab_size' : 30522, 'hidden_size' : 1024, 'num_hidden_layers' : 24, 'num_attention_heads' : 16, 
      'intermediate_size' : 4096, 'hidden_dropout_prob' : 0.1, 'attention_probs_dropout_prob' : 0.1,
      'max_position_embeddings' : 512, 'type_vocab_size' : 2
      }

    # Create in tinygrad
    mdl = BertForQuestionAnswering(**config)
    mdl.load_from_pretrained()

    # Create in torch
    with torch.no_grad():
      fn = Path(__file__).parent.parent.parent / "weights/bert_for_qa.pt"
      torch_mdl = TorchBertForQuestionAnswering.from_pretrained(fn, config=BertConfig(**config))

    seeds = [1337, 3141, 1602]
    bsz, seq_len = 1, 384
    for i in range(3):
      in_ids, mask, seg_ids = get_question_samp(bsz, seq_len, config['vocab_size'], seeds[i])
      out = mdl(Tensor(in_ids), Tensor(mask), Tensor(seg_ids))
      torch_out = torch_mdl.forward(torch.from_numpy(in_ids).long(), torch.from_numpy(mask), torch.from_numpy(seg_ids).long())[:2]
      torch_out = torch.cat(torch_out).unsqueeze(2)
      np.testing.assert_allclose(out.numpy(), torch_out.detach().numpy(), atol=5e-4, rtol=5e-5)

if __name__ == '__main__':
  unittest.main()
