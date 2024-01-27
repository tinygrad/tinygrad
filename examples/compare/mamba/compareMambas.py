from transformers import AutoTokenizer
from mambaTorch import Mamba as MambaTorch
from mambaTorch import generate as genTorch
from mambaTiny import Mamba as MambaTiny
from mambaTiny import generate as genTiny
from mambaSpeed import Mamba as MambaSpeed
from mambaSpeed import generate as genSpeed
import time
from tqdm import tqdm

torch_device='cuda'

PROMPT = 'Why is gravity '
NEW_TOKENS = 10
MODEL_SIZE = '370m'
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
modelTorch = MambaTorch.from_pretrained(MODEL_SIZE, device=torch_device)
modelTiny = MambaTiny.from_pretrained(MODEL_SIZE)
modelSpeed = MambaSpeed.from_pretrained(MODEL_SIZE)

s = time.time()
outTorch = genTorch(modelTorch, tokenizer, PROMPT, NEW_TOKENS, device=torch_device)
print(f"torch: {time.time() - s}")
s = time.time()
outSpeed = genSpeed(modelSpeed, tokenizer, PROMPT, NEW_TOKENS)
print(f"speed: {time.time() - s}")
s = time.time()
outTiny = genTiny(modelTiny, tokenizer, PROMPT, NEW_TOKENS)
print(f"tiny: {time.time() - s}")


assert outTorch == outTiny
assert outTorch == outSpeed
assert outTiny == outSpeed
print(outTiny)




