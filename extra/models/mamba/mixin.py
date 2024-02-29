# from collections import namedtuple
# from dataclasses import dataclass, field
# from typing import Optional
# import torch
# import torch.nn.functional as F
# from torch import Tensor
# from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput, TextStreamer


# @dataclass
# class InferenceParams:
#     """Inference parameters that are passed to the main model in order
#     to efficienly calculate and store the context during inference."""

#     seqlen_offset: int = 0
#     key_value_memory_dict: dict = field(default_factory=dict)
#     lengths_per_sample: Optional[Tensor] = None

#     def reset(self):
#         self.seqlen_offset = 0
#         if self.lengths_per_sample is not None:
#             self.lengths_per_sample.zero_()


# # https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/sampling.py
# # https://github.com/huggingface/transformers/blob/a44985b41cfa2de48a5e1de7f1f93b7483da25d1/src/transformers/generation/logits_process.py#L231
# def modify_logits_for_top_k_filtering(logits, top_k):
#     """Set the logits for none top-k values to -inf. Done in-place."""
#     indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
#     logits.masked_fill_(indices_to_remove, float("-Inf"))


# # https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/sampling.py
# # https://github.com/huggingface/transformers/blob/a44985b41cfa2de48a5e1de7f1f93b7483da25d1/src/transformers/generation/logits_process.py#L170
# def modify_logits_for_top_p_filtering(logits, top_p):
#     """Set the logits for none top-p values to -inf. Done in-place."""
#     if top_p <= 0.0 or top_p >= 1.0:
#         return
#     # First sort and calculate cumulative sum of probabilities.
#     sorted_logits, sorted_indices = torch.sort(logits, descending=False)
#     cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
#     # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
#     sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
#     # scatter sorted tensors to original indexing
#     indices_to_remove = sorted_indices_to_remove.scatter(
#         0, sorted_indices, sorted_indices_to_remove
#     )
#     logits.masked_fill_(indices_to_remove, float("-inf"))


# def modify_logit_for_repetition_penalty(logits, prev_output_tokens, repetition_penalty=1.0):
#     """Apply repetition penalty. See https://arxiv.org/abs/1909.05858
#     logits: (vocab_size)
#     prev_output_tokens: (seq_len)
#     """
#     if repetition_penalty == 1.0:
#         return logits
#     score = torch.gather(logits, 0, prev_output_tokens)
#     # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
#     score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
#     logits.scatter_(0, prev_output_tokens, score)
#     return logits


# def sample(logits, top_k=1, top_p=0.0, temperature=1.0):
#     """Sample from top-k logits.
#     Arguments:
#         logits: Tensor of shape (vocab_size)
#     """
#     if top_k == 1:  # Short-circuit for greedy decoding
#         return logits.argmax(dim=-1)
#     else:
#         if top_p > 0.0:
#             assert top_p <= 1.0, "top-p should be in (0, 1]."
#         if top_k > 0:
#             top_k = min(top_k, logits.size(-1))  # Safety check
#             logits_top, indices = torch.topk(logits, top_k, dim=-1)
#             if temperature != 1.0:
#                 logits_top /= temperature
#             modify_logits_for_top_p_filtering(logits_top, top_p)
#             return indices[
#                 torch.arange(indices.shape[0], device=indices.device),
#                 torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(dim=-1),
#             ]
#         else:
#             # Clone so that when we modify for top_p we don't change the original logits
#             logits_top = logits / temperature if temperature != 1.0 else logits.clone()
#             modify_logits_for_top_p_filtering(logits_top, top_p)
#             return torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(dim=-1)


# @torch.inference_mode()
# def decode(
#     input_ids,
#     model,
#     max_length,
#     top_k=1,
#     top_p=0.0,
#     temperature=1.0,
#     repetition_penalty=1.0,
#     eos_token_id=None,
#     vocab_size=None,
#     cg=False,
#     enable_timing=False,
#     streamer: Optional[TextStreamer] = None
# ):
#     """Decoding, either greedy or with top-k or top-p sampling.
#     If top-k = 0, don't limit the number of candidates (pure sampling).
#     Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
#     then top-p.

#     Arguments:
#         input_ids: (seq_len)
#         max_length: int
#     Returns: GreedySearchDecoderOnlyOutput or SampleDecoderOnlyOutput, with the following fields:
#         sequence: (max_length)
#         scores: tuples of (vocab_size)
#     """
#     if streamer is not None:
#         streamer.put(input_ids.cpu())

#     seqlen_og = input_ids.shape
#     inference_params = InferenceParams()

#     def get_logits(input_id):
#         logits = model(input_id, inference_params=inference_params).logits.squeeze(dim=1)
#         return logits[..., :vocab_size] if vocab_size is not None else logits

#     def should_stop(current_token):
#         if inference_params.seqlen_offset == 0:
#             return False
#         if eos_token_id is not None and (current_token == eos_token_id).all():
#             return True
#         if inference_params.seqlen_offset >= max_length - 1:
#             return True
#         return False

#     scores, sequence = [], [input_ids[0:1]]
#     sequence_cat = input_ids
#     while not should_stop(sequence[-1]):
#         logits = get_logits(sequence[-1])
#         inference_params.seqlen_offset += 1
#         if inference_params.seqlen_offset < input_ids.shape[0]:
#             sampled_token = input_ids[inference_params.seqlen_offset:inference_params.seqlen_offset+1]
#         elif repetition_penalty == 1.0:
#             sampled_token = sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)
#         else:
#             logits = modify_logit_for_repetition_penalty(
#                 logits.clone(), sequence_cat, repetition_penalty
#             )
#             sampled_token = sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)
#             sequence_cat = torch.cat([sequence_cat, sampled_token], dim=0)
#         sequence.append(sampled_token)
#         scores.append(logits)
#         if streamer is not None:
#             streamer.put(sampled_token.cpu())
#     if streamer is not None:
#         streamer.end()
#     output_cls = GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput
#     return output_cls(sequences=torch.cat(sequence, dim=0), scores=tuple(scores))


# class GenerationMixin:
#     def generate(
#         self,
#         input_ids,
#         max_length,
#         top_k=1,
#         top_p=0.0,
#         temperature=1.0,
#         return_dict_in_generate=False,
#         output_scores=False,
#         **kwargs,
#     ):
#         output = decode(input_ids, self, max_length, top_k=top_k, top_p=top_p, temperature=temperature, **kwargs)
#         if not output_scores:
#             output.scores = None
#         return output if return_dict_in_generate else output.sequence