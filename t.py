import os
model_path="tiny"
x = any(map(lambda f: f.endswith(".safetensors"), os.listdir(model_path)))
print(x)
