#!/usr/bin/env python3
import time
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import Counter

from tinygrad import Device, GlobalCounters, Tensor, TinyJit
from tinygrad.helpers import getenv, diskcache_clear, Context
from tinygrad.nn.state import get_parameters, load_state_dict, safe_load, safe_save
from tinygrad.nn.optim import AdamW

from extra.models.llama import Transformer, convert_from_huggingface, fix_bf16
from examples.mlperf.helpers import get_training_state
from examples.mlperf.llama2_70b_lora.lora import LoRAConfig, apply_lora_to_model, LoRAParameterManager
# Hugging Face model URLs
LLAMA_MODEL_URLS = {
    "7B": "https://huggingface.co/meta-llama/Llama-2-7b-hf/resolve/main/pytorch_model.bin",
    "70B": "https://huggingface.co/meta-llama/Llama-2-70b-hf/resolve/main/pytorch_model.bin"
}

from examples.mlperf.llama2_70b_lora.dataset import create_data_loaders, get_tokenizer

try:
    from mlperf_logging import mllog
    import mlperf_logging.mllog.constants as mllog_constants
    MLPERF_LOGGING_AVAILABLE = True
except ImportError:
    MLPERF_LOGGING_AVAILABLE = False


def compute_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, Dict[str, float]]:
    def tokenize(text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    def rouge_n(pred_tokens: List[str], ref_tokens: List[str], n: int) -> Dict[str, float]:
        pred_ngrams = get_ngrams(pred_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)
        
        if not ref_ngrams:
            return {"precision": 0.0, "recall": 0.0, "f": 0.0}
        
        overlap = sum((pred_ngrams & ref_ngrams).values())
        precision = overlap / max(sum(pred_ngrams.values()), 1)
        recall = overlap / sum(ref_ngrams.values())
        f_score = (2 * precision * recall) / max(precision + recall, 1e-8)
        
        return {"precision": precision, "recall": recall, "f": f_score}
    
    def rouge_l(pred_tokens: List[str], ref_tokens: List[str]) -> Dict[str, float]:
        def lcs_length(x: List[str], y: List[str]) -> int:
            m, n = len(x), len(y)
            if m == 0 or n == 0:
                return 0
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        
        if not ref_tokens:
            return {"precision": 0.0, "recall": 0.0, "f": 0.0}
        
        lcs = lcs_length(pred_tokens, ref_tokens)
        precision = lcs / max(len(pred_tokens), 1)
        recall = lcs / len(ref_tokens)
        f_score = (2 * precision * recall) / max(precision + recall, 1e-8)
        
        return {"precision": precision, "recall": recall, "f": f_score}
    
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize(pred)
        ref_tokens = tokenize(ref)
        
        rouge_1_scores.append(rouge_n(pred_tokens, ref_tokens, 1))
        rouge_2_scores.append(rouge_n(pred_tokens, ref_tokens, 2))
        rouge_l_scores.append(rouge_l(pred_tokens, ref_tokens))
    
    def avg_scores(scores: List[Dict[str, float]]) -> Dict[str, float]:
        if not scores:
            return {"precision": 0.0, "recall": 0.0, "f": 0.0}
        return {
            "precision": sum(s["precision"] for s in scores) / len(scores),
            "recall": sum(s["recall"] for s in scores) / len(scores),
            "f": sum(s["f"] for s in scores) / len(scores)
        }
    
    return {
        "rouge-1": avg_scores(rouge_1_scores),
        "rouge-2": avg_scores(rouge_2_scores),
        "rouge-l": avg_scores(rouge_l_scores)
    }


@dataclass
class TrainingConfig:
    gpus: List[str]
    seed: int = 42
    batch_size: int = 1
    base_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_length: int = 8192
    target_rouge: float = 0.270
    max_steps: int = 50000
    eval_step_interval: int = 500
    ckpt_step_interval: int = 500
    max_eval_batches: int = 100
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_target_modules: List[str] = None
    basedir: Path = Path("./")
    datadir: Path = Path("./dataset/govreport")
    modeldir: Path = Path("./models/llama-2-70b")
    ckptdir: Path = Path("./checkpoints")
    resume_ckptdir: str = ""
    resume_itr: int = 0
    wandb_enabled: bool = False
    wandb_resume_id: str = ""
    mlperf_logging: bool = False
    init_mlperf: bool = False
    run_mlperf: bool = False
    device: str = 'auto'
    num_epochs: int = 3

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["wq", "wv", "wk", "wo"]
        if self.resume_itr or self.resume_ckptdir:
            assert self.resume_itr and self.resume_ckptdir, "Both resume_itr and resume_ckptdir must be set"

    @property
    def learning_rate(self) -> float:
        return self.batch_size * self.base_learning_rate

    @classmethod
    def from_env(cls) -> 'TrainingConfig':
        num_gpus = getenv("GPUS", 1)
        gpus = [f"{Device.DEFAULT}:{i}" for i in range(num_gpus)]
        
        return cls(
            gpus=gpus,
            seed=getenv("SEED", 42),
            batch_size=getenv("BS", 1 * len(gpus)),
            base_learning_rate=getenv("LEARNING_RATE", 1e-4),
            weight_decay=getenv("WEIGHT_DECAY", 0.01),
            max_length=getenv("MAX_LENGTH", 8192),
            target_rouge=getenv("TARGET_ROUGE", 0.270),
            max_steps=getenv("MAX_STEPS", 50000),
            eval_step_interval=getenv("EVAL_STEP_INTERVAL", 500),
            ckpt_step_interval=getenv("CKPT_STEP_INTERVAL", 500),
            lora_r=getenv("LORA_R", 16),
            lora_alpha=getenv("LORA_ALPHA", 32.0),
            lora_target_modules=getenv("LORA_TARGET_MODULES", "wq,wv,wk,wo").split(','),
            basedir=Path(getenv("BASEDIR", "./")),
            datadir=Path(getenv("DATADIR", "./dataset/govreport")),
            modeldir=Path(getenv("MODELDIR", "./models/llama-2-70b")),
            ckptdir=Path(getenv("CKPTDIR", "./checkpoints")),
            resume_ckptdir=getenv("RESUME_CKPTDIR", ""),
            resume_itr=getenv("RESUME_ITR", 0),
            wandb_enabled=bool(getenv("WANDB")),
            wandb_resume_id=getenv("WANDB_RESUME", ""),
            mlperf_logging=bool(getenv("LOGMLPERF")),
            init_mlperf=bool(getenv("INITMLPERF")),
            run_mlperf=bool(getenv("RUNMLPERF")),
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'TrainingConfig':
        if args.device == 'cpu':
            gpus = ['CPU:0']
        else:
            gpus = [f"{Device.DEFAULT}:{i}" for i in range(args.gpus)]
        
        return cls(
            gpus=gpus,
            seed=args.seed,
            batch_size=args.batch_size,
            base_learning_rate=args.learning_rate,
            max_length=args.max_length,
            target_rouge=args.target_rouge,
            eval_step_interval=args.eval_steps,
            ckpt_step_interval=args.save_steps,
            num_epochs=args.num_epochs,
            datadir=Path(args.dataset_path),
            modeldir=Path(args.model_path),
            ckptdir=Path(args.output_dir),
            device=args.device,
        )


class MLPerfLogger:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = None
        
        if config.mlperf_logging:
            if not MLPERF_LOGGING_AVAILABLE:
                raise RuntimeError("MLPerf logging requested but not available")
            
            mllog.config(filename=f"result_llama2_lora_{config.seed}.txt")
            mllog.config(root_dir=Path(__file__).parents[3].as_posix())
            self.logger = mllog.get_mllogger()

    def init_start(self):
        if self.logger and self.config.init_mlperf:
            self.logger.event(key=mllog_constants.SUBMISSION_ORG, value="tinycorp")
            self.logger.event(key=mllog_constants.SUBMISSION_PLATFORM, value=getenv("SUBMISSION_PLATFORM", "tinybox"))
            self.logger.event(key=mllog_constants.SUBMISSION_DIVISION, value=mllog_constants.CLOSED)
            self.logger.event(key=mllog_constants.SUBMISSION_STATUS, value=mllog_constants.ONPREM)
            self.logger.event(key="submission_benchmark", value="llama2_70b_lora")
            diskcache_clear()
            self.logger.event(key=mllog_constants.CACHE_CLEAR, value=True)
            self.logger.start(key=mllog_constants.INIT_START)

    def init_end(self):
        if self.logger:
            self.logger.end(key=mllog_constants.INIT_END)

    def run_start(self):
        if self.logger and self.config.run_mlperf:
            self.logger.start(key=mllog_constants.RUN_START)
            self.logger.event(key=mllog_constants.SEED, value=self.config.seed)

    def run_stop(self, status: str, step: int):
        if self.logger:
            self.logger.end(key=mllog_constants.RUN_STOP, metadata={"status": status, "step": step})

    def log_eval_rouge(self, rouge_score: float, step: int):
        if self.logger:
            self.logger.event(key="eval_rouge_l_f1", value=rouge_score, metadata={"step": step})


class ModelManager:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.lora_params = None

    def setup_model(self) -> Transformer:
        print("Loading Llama2 70B model...")
        
        llama_config = {
            "dim": 8192,
            "hidden_dim": 28672,
            "n_heads": 64,
            "n_kv_heads": 8,
            "n_layers": 80,
            "norm_eps": 1e-5,
            "vocab_size": 32000,
            "max_context": self.config.max_length,
            "jit": False
        }
        
        self.model = Transformer(**llama_config)
        self._load_weights()
        self._apply_lora()
        self._shard_model()
        
        return self.model

    def _load_weights(self):
        if not Path(self.config.modeldir).exists():
            print(f"Warning: Model directory {self.config.modeldir} not found, using random weights")
            return

        mp = Path(self.config.modeldir)
        if mp.is_dir():
            st_single = mp / "model.safetensors"
            st_index = mp / "model.safetensors.index.json"
            
            if st_single.exists():
                weights = safe_load(st_single)
            elif st_index.exists():
                with open(st_index, "r") as f:
                    index = json.load(f)
                weight_map = index.get("weight_map", {})
                shard_files = sorted({mp / fname for fname in weight_map.values()})
                shard_state = {}
                for sf in shard_files:
                    shard_state[str(sf.name)] = safe_load(sf)
                weights = {name: shard_state[weight_map[name]][name] for name in weight_map.keys()}
            else:
                return
        else:
            weights = safe_load(mp)

        weights = fix_bf16(weights)
        if any('model.layers' in k for k in weights.keys()):
            weights = convert_from_huggingface(weights, 80, 64, 8)
        load_state_dict(self.model, weights)
        print(f"Loaded model weights from {self.config.modeldir}")

    def _apply_lora(self):
        print("Applying LoRA adapters...")
        lora_config = LoRAConfig(
            r=self.config.lora_r,
            alpha=self.config.lora_alpha,
            dropout=0.1,
            target_modules=self.config.lora_target_modules
        )
        
        apply_lora_to_model(model=self.model, config=lora_config)
        LoRAParameterManager.freeze_base_model(model=self.model)
        self.lora_params = LoRAParameterManager.get_lora_parameters(model=self.model)
        print(f"LoRA parameters: {len(self.lora_params)}")

    def _shard_model(self):
        if len(self.config.gpus) > 1:
            to_move = get_parameters(self.model)
            for p in to_move:
                p.to_(self.config.gpus)
            with Context(BEAM=0):
                Tensor.realize(*to_move)

    def get_lora_parameters(self) -> List[Tensor]:
        return self.lora_params


@TinyJit
def train_step(input_ids: Tensor, labels: Tensor, model: Transformer, optimizer: AdamW) -> Tensor:
    optimizer.zero_grad()
    
    logits = model.forward(input_ids, start_pos=0, temperature=float('nan'))
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    shift_logits_flat = shift_logits.reshape(-1, shift_logits.shape[-1])
    shift_labels_flat = shift_labels.reshape(-1)
    
    valid_mask = shift_labels_flat != -100
    if valid_mask.sum() == 0:
        return Tensor([0.0])
    
    num_classes = shift_logits_flat.shape[-1]
    log_probs = shift_logits_flat.log_softmax(axis=-1)
    one_hot_labels = shift_labels_flat.one_hot(num_classes)
    per_token_loss = -(one_hot_labels * log_probs).sum(axis=-1)
    per_token_loss_masked = per_token_loss * valid_mask.cast(per_token_loss.dtype)
    loss = per_token_loss_masked.sum() / valid_mask.sum().cast(per_token_loss.dtype)
    
    loss.backward()
    optimizer.step()
    
    loss_cpu = loss.detach().to("CPU")
    Tensor.realize(loss_cpu)
    return loss_cpu


@Tensor.train(mode=False)
def evaluate(model: Transformer, val_loader, tokenizer, max_eval_batches: int = 100) -> Tuple[float, Dict]:
    total_loss = 0.0
    num_batches = 0
    predictions = []
    references = []
    
    print(f"Evaluating on up to {max_eval_batches} batches...")
    
    for i, batch in enumerate(val_loader):
        if i >= max_eval_batches:
            break
        
        with Tensor.no_grad():
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            logits = model.forward(input_ids, start_pos=0, temperature=float('nan'))
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            shift_logits_flat = shift_logits.reshape(-1, shift_logits.shape[-1])
            shift_labels_flat = shift_labels.reshape(-1)
            
            valid_mask = shift_labels_flat != -100
            if valid_mask.sum() > 0:
                loss = shift_logits_flat[valid_mask].sparse_categorical_crossentropy(shift_labels_flat[valid_mask])
                total_loss += loss.item()
                num_batches += 1
            
            pred_ids = logits.argmax(axis=-1).numpy()
            pred_text = tokenizer.decode(pred_ids[0].tolist())
            
            ref_ids = labels[0][labels[0] != -100].numpy().tolist()
            ref_text = tokenizer.decode(ref_ids)
            
            predictions.append(pred_text)
            references.append(ref_text)
    
    avg_loss = total_loss / max(num_batches, 1)
    rouge_scores = compute_rouge_scores(predictions, references)
    
    return avg_loss, rouge_scores


class CheckpointManager:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.config.ckptdir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, model: Transformer, optimizer: AdamW, step: int, prefix: str = "ckpt"):
        print(f"Saving checkpoint: {prefix}_{step}")
        
        ckpt = get_training_state(model, optimizer, None)
        
        cpu_ckpt = {}
        for k, v in ckpt.items():
            v.realize()
            cpu_tensor = v.detach().to("CPU")
            cpu_tensor.realize()
            cpu_ckpt[k] = cpu_tensor.cast(cpu_tensor.dtype.base).contiguous()
        
        Tensor.realize(*[v for v in cpu_ckpt.values()])
        
        checkpoint_path = self.config.ckptdir / f"{prefix}_{step}.safetensors"
        safe_save(cpu_ckpt, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, model: Transformer, optimizer: AdamW):
        if not self.config.resume_ckptdir:
            return
        
        print(f"Resuming from {self.config.resume_ckptdir} at iteration {self.config.resume_itr}")
        ckpt = safe_load(f"{self.config.resume_ckptdir}/backup_{self.config.resume_itr}.safetensors")
        
        for obj, pat in [(model, "model."), (optimizer, "optimizer.")]:
            sd = {k.split(pat)[1]: v for k, v in ckpt.items() if k.startswith(pat)}
            with Context(DEBUG=1):
                load_state_dict(obj, sd, strict=False)


def train_llama2_lora(config: TrainingConfig):
    print(f"Training on {config.gpus}")
    print(f"BS={config.batch_size}, BASE_LR={config.base_learning_rate}, lr={config.learning_rate}")
    
    for device in config.gpus:
        Device[device]
    
    Tensor.manual_seed(config.seed)
    
    mlperf_logger = MLPerfLogger(config)
    mlperf_logger.init_start()
    
    model_manager = ModelManager(config)
    model = model_manager.setup_model()
    lora_params = model_manager.get_lora_parameters()
    
    optimizer = AdamW(lora_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    
    checkpoint_manager = CheckpointManager(config)
    checkpoint_manager.load_checkpoint(model, optimizer)
    
    tokenizer = get_tokenizer()
    train_loader, val_loader = create_data_loaders(
        data_dir=str(config.datadir),
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_length
    )
    
    mlperf_logger.init_end()
    mlperf_logger.run_start()
    
    print("Starting training...")
    best_rouge = 0.0
    achieved_target = False
    global_step = 0
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        model.train()
        
        for batch in train_loader:
            global_step += 1
            
            if config.max_steps and global_step > config.max_steps:
                break
            
            GlobalCounters.reset()
            t1 = time.perf_counter()
            
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            if len(config.gpus) > 1:
                input_ids.shard_(config.gpus, axis=0)
                labels.shard_(config.gpus, axis=0)
            
            loss = train_step(input_ids, labels, model, optimizer)
            loss_item = loss.item()
            
            t2 = time.perf_counter()
            
            if global_step % 10 == 0:
                gflops = GlobalCounters.global_ops * 1e-9 / (t2-t1)
                print(f"step {global_step}: {gflops:9.2f} GFLOPS, loss: {loss_item:.5f}")
            
            if global_step % config.eval_step_interval == 0:
                print(f"\nEvaluating at step {global_step}...")
                eval_loss, rouge_scores = evaluate(model, val_loader, tokenizer, config.max_eval_batches)
                rouge_l_f = rouge_scores.get('rouge-l', {}).get('f', 0.0)
                
                print(f"Eval - Loss: {eval_loss:.4f}, ROUGE-L F1: {rouge_l_f:.4f}")
                
                mlperf_logger.log_eval_rouge(rouge_l_f, global_step)
                
                if rouge_l_f >= config.target_rouge and not achieved_target:
                    print(f"🎉 Target ROUGE-L {config.target_rouge} achieved! ({rouge_l_f:.4f})")
                    achieved_target = True
                    best_rouge = rouge_l_f
                    mlperf_logger.run_stop("success", global_step)
                    checkpoint_manager.save_checkpoint(model, optimizer, global_step, "final")
                    return
                
                if rouge_l_f > best_rouge:
                    best_rouge = rouge_l_f
                    checkpoint_manager.save_checkpoint(model, optimizer, global_step, "best")
            
            if global_step % config.ckpt_step_interval == 0:
                checkpoint_manager.save_checkpoint(model, optimizer, global_step)
        
        if achieved_target or (config.max_steps and global_step >= config.max_steps):
            break
    
    print(f"\nTraining completed!")
    print(f"Best ROUGE-L: {best_rouge:.4f}")
    print(f"Target achieved: {achieved_target}")
    
    if not achieved_target:
        mlperf_logger.run_stop("aborted", global_step)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='MLPerf Llama2 70B LoRA Training')
    parser.add_argument('--model_path', type=str, required=True, help='Path to Llama2 70B model weights')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to GovReport dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--max_length', type=int, default=8192, help='Maximum sequence length')
    parser.add_argument('--eval_steps', type=int, default=500, help='Evaluation frequency')
    parser.add_argument('--save_steps', type=int, default=1000, help='Save frequency')
    parser.add_argument('--target_rouge', type=float, default=0.270, help='Target ROUGE-L score')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu'], 
                       help='Device to use for training')
    parser.add_argument('--require_dataset', action='store_true', 
                       help='Fail if GovReport dataset files are missing')
    return parser


def main():
    if getenv("GPUS"):
        config = TrainingConfig.from_env()
        train_llama2_lora(config)
    else:
        parser = create_argument_parser()
        args = parser.parse_args()
        config = TrainingConfig.from_args(args)
        train_llama2_lora(config)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    main()