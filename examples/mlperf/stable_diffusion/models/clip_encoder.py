from transformers import CLIPTokenizer, CLIPTextModel

# Define a function to encode text using CLIP model 
def encode_text(text):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    return text_encoder(**inputs).last_hidden_state