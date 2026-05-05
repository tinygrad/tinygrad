import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import extra.torch_backend.backend  # type: ignore # noqa

MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

model_torch = AutoModelForCausalLM.from_pretrained(MODEL)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt")

generated_ids = model_torch.generate(
    **model_inputs,
    max_new_tokens=200,
    do_sample=False,
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

model_tiny = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to("tiny")

print(type(model_tiny))
generated_ids_tiny = model_tiny.generate(
    **model_inputs.to("tiny"),
    max_new_tokens=10,
    do_sample=False,
)
generated_ids_tiny = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids_tiny)
]

response_tiny = tokenizer.batch_decode(generated_ids_tiny, skip_special_tokens=True)[0]

print(response_tiny)
