import json
import random
from pathlib import Path
from typing import Dict, List, Iterator
from tinygrad import Tensor

LABEL_IGNORE_INDEX = -100
INSTRUCTION_PROMPT_TEMPLATE = "Summarize the following government document:\n\n{input}\n\nSummary:"

class GovReportExample:
    def __init__(self, input_text: str, output_text: str, example_id: str | int):
        self.input_text = input_text
        self.output_text = output_text
        self.example_id = example_id

class GovReportDataset:
    def __init__(self, data_path: str, tokenizer, max_length: int = 8192, split: str = "train"):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.examples = self._load_examples()
        print(f"Loaded {len(self.examples)} examples for {split} split")
    
    def _load_examples(self) -> List[GovReportExample]:
        split_file = self.data_path / f"{self.split}.json"
        
        if not split_file.exists():
            self._create_dummy_data()
        
        with open(split_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        return [
            GovReportExample(
                input_text=item['input'],
                output_text=item['output'],
                example_id=item.get('id', idx)
            )
            for idx, item in enumerate(raw_data)
        ]
    
    def _create_dummy_data(self):
        self.data_path.mkdir(parents=True, exist_ok=True)
        dummy_examples = [
            {
                "input": "This is a sample government report about policy implementation. " * 50,
                "output": "This report discusses policy implementation challenges and recommendations.",
                "id": f"dummy_{i}"
            }
            for i in range(10)
        ]
        
        for split_name in ["train", "validation", "test"]:
            output_file = self.data_path / f"{split_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dummy_examples, f, ensure_ascii=False, indent=2)
        print(f"Created dummy data: {len(dummy_examples)} examples per split")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> GovReportExample:
        return self.examples[idx]
    
    def _tokenize_example(self, example: GovReportExample) -> Dict[str, List[int]]:
        prompt = INSTRUCTION_PROMPT_TEMPLATE.format(input=example.input_text)
        
        # The SimpleTokenizer.encode() method doesn't take add_special_tokens parameter
        # We need to manually add special tokens
        input_tokens = [self.tokenizer.bos_token_id] + self.tokenizer.encode(prompt)
        target_tokens = self.tokenizer.encode(example.output_text)
        
        combined_tokens = input_tokens + target_tokens + [self.tokenizer.eos_token_id]
        
        if len(combined_tokens) > self.max_length:
            combined_tokens = combined_tokens[:self.max_length]
        
        input_length = len(input_tokens)
        label_tokens = [LABEL_IGNORE_INDEX] * input_length + target_tokens + [self.tokenizer.eos_token_id]
        
        if len(label_tokens) > self.max_length:
            label_tokens = label_tokens[:self.max_length]
        
        actual_length = min(len(combined_tokens), len(label_tokens))
        attention_mask = [1] * actual_length + [0] * (self.max_length - actual_length)
        
        combined_tokens += [self.tokenizer.pad_token_id] * (self.max_length - len(combined_tokens))
        label_tokens += [LABEL_IGNORE_INDEX] * (self.max_length - len(label_tokens))
        
        return {
            'input_ids': combined_tokens,
            'attention_mask': attention_mask,
            'labels': label_tokens
        }
    
    def collate_fn(self, batch: List[GovReportExample]) -> Dict[str, Tensor]:
        tokenized_batch = [self._tokenize_example(example) for example in batch]
        
        return {
            'input_ids': Tensor([item['input_ids'] for item in tokenized_batch], dtype='int32'),
            'attention_mask': Tensor([item['attention_mask'] for item in tokenized_batch], dtype='int32'),
            'labels': Tensor([item['labels'] for item in tokenized_batch], dtype='int32')
        }

class GovReportDataLoader:
    def __init__(self, dataset: GovReportDataset, batch_size: int = 1, shuffle: bool = True, drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = list(range(len(dataset)))
    
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[Dict[str, Tensor]]:
        if self.shuffle:
            random.shuffle(self.indices)
        
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            
            batch = [self.dataset[idx] for idx in batch_indices]
            yield self.dataset.collate_fn(batch)

def create_data_loaders(data_dir: str, tokenizer, batch_size: int = 1, max_length: int = 8192):
    train_dataset = GovReportDataset(
        data_path=data_dir, 
        tokenizer=tokenizer, 
        max_length=max_length, 
        split="train"
    )
    val_dataset = GovReportDataset(
        data_path=data_dir, 
        tokenizer=tokenizer, 
        max_length=max_length, 
        split="validation"
    )
    
    train_loader = GovReportDataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    val_loader = GovReportDataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    return train_loader, val_loader

def get_tokenizer(model_path: str = None):
    from tinygrad import Tensor
    from tinygrad.apps.llm import SimpleTokenizer
    
    if model_path and Path(model_path).exists():
        # Load tokenizer from the actual model being fine-tuned
        if model_path.endswith('.gguf'):
            model_gguf = Tensor(model_path)
        else:
            # For safetensors/bin files, we need to look for tokenizer files
            model_dir = Path(model_path).parent if Path(model_path).is_file() else Path(model_path)
            tokenizer_model = model_dir / "tokenizer.model"
            if tokenizer_model.exists():
                # Load from sentencepiece model if available
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor()
                sp.load(str(tokenizer_model))
                # Create a wrapper that mimics SimpleTokenizer interface
                class SentencePieceTokenizer:
                    def __init__(self, sp_model):
                        self.sp = sp_model
                        self.pad_token_id = sp_model.pad_id()
                        self.bos_token_id = sp_model.bos_id() 
                        self.eos_token_id = sp_model.eos_id()
                    
                    def encode(self, text: str) -> list[int]:
                        return self.sp.encode(text, out_type=int)
                    
                    def decode(self, ids: list[int]) -> str:
                        return self.sp.decode(ids)
                
                return SentencePieceTokenizer(sp)
            else:
                raise FileNotFoundError(f"No tokenizer found in {model_dir}")
    else:
        # Fallback: use Llama2 tokenizer from a Llama2 GGUF model
        # Use Llama2 7B as it has the same tokenizer as 70B but is smaller to download
        llama2_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q2_K.gguf"
        print(f"Loading Llama2 tokenizer from {llama2_url}")
        model_gguf = Tensor.from_url(llama2_url)
    
    # Load tokenizer from GGUF
    from tinygrad.nn.state import gguf_load
    kv, _ = gguf_load(model_gguf.to(None))
    
    # Create the tokenizer from GGUF key-values
    tokenizer = SimpleTokenizer.from_gguf_kv(kv)
    
    # Add compatibility attributes for the training code
    tokenizer.pad_token_id = kv.get('tokenizer.ggml.padding_token_id', 0)
    tokenizer.bos_token_id = kv.get('tokenizer.ggml.bos_token_id', 1)
    tokenizer.eos_token_id = kv.get('tokenizer.ggml.eos_token_id', 2)
    
    return tokenizer