# import argparse
# import time
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from mamba_simple import MambaLMHeadModel


# parser = argparse.ArgumentParser(description="Text Generation")
# parser.add_argument("--model-name", type=str, default="state-spaces/mamba-130m")
# parser.add_argument("--prompt", type=str, default=None)
# parser.add_argument("--promptlen", type=int, default=100)
# parser.add_argument("--genlen", type=int, default=100)
# parser.add_argument("--temperature", type=float, default=1.0)
# parser.add_argument("--topk", type=int, default=1)
# parser.add_argument("--topp", type=float, default=1.0)
# parser.add_argument("--repetition-penalty", type=float, default=1.0)
# args = parser.parse_args()

# device = "cpu"
# dtype = torch.float32

# print(f"Loading model {args.model_name}")
# is_mamba = args.model_name.startswith("state-spaces/mamba-")
# if is_mamba:
#     tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
#     model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=dtype)
# else:
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name)
#     model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map={"": device}, torch_dtype=dtype)
# model.eval()
# print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# start = time.time()

# torch.random.manual_seed(0)
# if args.prompt is None:
#     input_ids = torch.randint(1, 1000, (1, args.promptlen), dtype=torch.long)
#     attn_mask = torch.ones_like(input_ids, dtype=torch.long)
# else:
#     tokens = tokenizer(args.prompt, return_tensors="pt")
#     input_ids = tokens.input_ids.to(device=device)
#     attn_mask = tokens.attention_mask.to(device=device)
# max_length = input_ids.shape[1] + args.genlen

# if is_mamba:
#     input_ids = input_ids[0]  # remove the batch dimension
#     fn = lambda: model.generate(
#         input_ids=input_ids,
#         max_length=max_length,
#         cg=True,
#         return_dict_in_generate=True,
#         output_scores=True,
#         enable_timing=False,
#         temperature=args.temperature,
#         top_k=args.topk,
#         top_p=args.topp,
#         repetition_penalty=args.repetition_penalty,
#     )
# else:
#     fn = lambda: model.generate(
#         input_ids=input_ids,
#         attention_mask=attn_mask,
#         max_length=max_length,
#         return_dict_in_generate=True,
#         pad_token_id=tokenizer.eos_token_id,
#         do_sample=True,
#         temperature=args.temperature,
#         top_k=args.topk,
#         top_p=args.topp,
#         repetition_penalty=args.repetition_penalty,
#     )
# out = fn()
# if args.prompt is not None:
#     print(''.join(tokenizer.batch_decode(out.sequences)))

# print(f"Prompt length: {len(input_ids)}, generation length: {len(out.sequences) - len(input_ids)}")
# print(f"{args.model_name} prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms")


