from typing import Dict, List, Optional, Union
from tinygrad import Tensor, nn
from tinygrad.helpers import getenv
from dataclasses import dataclass

@dataclass
class LoRAConfig:
    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.1
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["wq", "wv", "wk", "wo"]

class LoRALinear:
    def __init__(self, in_features: int, out_features: int, r: int = 16, alpha: float = 32.0, dropout: float = 0.1, bias: bool = False):
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0
        self.enabled = True
        self.merged = False
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.lora_B.weight.assign(Tensor.zeros_like(self.lora_B.weight))
            self.dropout_p = dropout
            self.dropout = None
    
    def __call__(self, x: Tensor) -> Tensor:
        if self.r == 0 or not self.enabled:
            return self.linear(x)
        
        base_output = self.linear(x)
        
        if self.merged:
            return base_output
        
        lora_output = self.lora_A(x)
        if self.dropout_p > 0.0:
            mask = Tensor.rand(*lora_output.shape) > self.dropout_p
            lora_output = lora_output * mask / (1.0 - self.dropout_p)
        lora_output = self.lora_B(lora_output) * self.scaling
        
        return base_output + lora_output
    
    def merge_weights(self):
        if self.r > 0 and not self.merged:
            merged_weight = self.linear.weight + (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.linear.weight.assign(merged_weight)
            self.merged = True
    
    def unmerge_weights(self):
        if self.r > 0 and self.merged:
            merged_weight = self.linear.weight - (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.linear.weight.assign(merged_weight)
            self.merged = False

class LoRATransformerAdapter:
    def __init__(self, r: int = 16, alpha: float = 32.0, dropout: float = 0.1, target_modules: List[str] = None, layers_to_transform: Optional[List[int]] = None):
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["wq", "wv", "wk", "wo"]
        self.layers_to_transform = layers_to_transform
        self.applied_layers = {}
    
    def apply(self, model) -> Dict[str, Dict[str, Dict[str, nn.Linear]]]:
        original_modules = {}
        
        for layer_idx, layer in enumerate(model.layers):
            if self.layers_to_transform is not None and layer_idx not in self.layers_to_transform:
                continue
            
            layer_key = f"layer_{layer_idx}"
            original_modules[layer_key] = {}
            
            for module_name in self.target_modules:
                if hasattr(layer.attention, module_name):
                    original_linear = getattr(layer.attention, module_name)
                    lora_linear = self._create_lora_linear(original_linear)
                    setattr(layer.attention, module_name, lora_linear)
                    
                    if layer_key not in original_modules:
                        original_modules[layer_key] = {}
                    original_modules[layer_key][module_name] = original_linear
        
        self.applied_layers = original_modules
        return original_modules
    
    def _create_lora_linear(self, original_linear: nn.Linear) -> LoRALinear:
        in_features = original_linear.weight.shape[1]
        out_features = original_linear.weight.shape[0]
        bias = original_linear.bias is not None
        
        lora_linear = LoRALinear(
            in_features=in_features,
            out_features=out_features,
            r=self.r,
            alpha=self.alpha,
            dropout=self.dropout,
            bias=bias
        )
        
        lora_linear.linear.weight.assign(original_linear.weight.detach())
        if bias and original_linear.bias is not None:
            lora_linear.linear.bias.assign(original_linear.bias.detach())
        
        return lora_linear

class LoRAParameterManager:
    @staticmethod
    def freeze_base_model(model):
        for layer in model.layers:
            for name in ["wq", "wv", "wk", "wo"]:
                if hasattr(layer.attention, name):
                    module = getattr(layer.attention, name)
                    if isinstance(module, LoRALinear):
                        module.linear.weight.requires_grad = False
                        if module.linear.bias is not None:
                            module.linear.bias.requires_grad = False
    
    @staticmethod
    def get_lora_parameters(model) -> List[Tensor]:
        lora_params = []
        for layer in model.layers:
            for name in ["wq", "wv", "wk", "wo"]:
                if hasattr(layer.attention, name):
                    module = getattr(layer.attention, name)
                    if isinstance(module, LoRALinear) and module.r > 0:
                        lora_params.append(module.lora_A.weight)
                        lora_params.append(module.lora_B.weight)
        return lora_params

def apply_lora_to_model(model, config: Optional[LoRAConfig] = None) -> Dict[str, Dict[str, Dict[str, nn.Linear]]]:
    if config is None:
        config = LoRAConfig()
    
    adapter = LoRATransformerAdapter(
        r=config.r,
        alpha=config.alpha,
        dropout=config.dropout,
        target_modules=config.target_modules
    )
    
    return adapter.apply(model=model)

def get_lora_config() -> Dict[str, Union[int, float, List[str]]]:
    return {
        'r': getenv("LORA_R", 16),
        'alpha': getenv("LORA_ALPHA", 32.0),
        'dropout': getenv("LORA_DROPOUT", 0.1),
        'target_modules': getenv("LORA_TARGET_MODULES", "wq,wv,wk,wo").split(','),
    }