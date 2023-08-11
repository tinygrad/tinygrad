#!/usr/bin/env python
import unittest
from pathlib import Path
import numpy as np
from tinygrad.tensor import Tensor
from extra.datasets.squad import iterate
import torch

class TestBert(unittest.TestCase):
  def test_questions(self):
    from models.bert import BertForQuestionAnswering
    from transformers import BertForQuestionAnswering as TorchBertForQuestionAnswering
    from transformers import BertTokenizer, BertConfig

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
    
    # Get samples
    tokenizer = BertTokenizer(str(Path(__file__).parent.parent.parent / "weights/bert_vocab.txt"))

    for _ in range(3):
      X, _ = next(iterate(tokenizer))
      for x in X:
        in_ids, mask, seg_ids = x["input_ids"], x["input_mask"], x["segment_ids"]
        out = mdl(Tensor(in_ids), Tensor(mask), Tensor(seg_ids))
        torch_out = torch_mdl.forward(torch.from_numpy(in_ids).long(), torch.from_numpy(mask), torch.from_numpy(seg_ids).long())[:2]
        torch_out = torch.cat(torch_out).unsqueeze(2)
        np.testing.assert_allclose(out.numpy(), torch_out.detach().numpy(), atol=1e-4, rtol=1e-5)

if __name__ == '__main__':
  unittest.main()
