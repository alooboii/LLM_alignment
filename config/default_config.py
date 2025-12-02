"""
Default Configuration for Alignment Methods Assignment
Covers: DPO, PPO (sparse/dense), GRPO, Reward Model, Evaluation, Analysis

This configuration file contains all hyperparameters and settings for:
- Data preparation
- Reward model training
- DPO training
- PPO training (sparse and dense reward)
- GRPO training
- Evaluation pipeline
- Analysis and metrics computation

Usage:
    python train_reward_model.py --batch_size 16 --lr 5e-5 --epochs 3 --seed 42
    python train_dpo.py --method DPO --lr 1e-4 --lora_r 8 --epochs 3 --seed 42
    python train_ppo.py --method PPO --reward_mode sparse --kl_coef 0.05 --seed 42
"""

import os
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
import json

# =============================================================================
# PROJECT PATHS
# =============================================================================

@dataclass
class PathConfig:
    """Directory structure for the project"""
    
    # Root directory
    project_root: Path = Path(__file__).parent
    
    # Data directories
    data_root: Path = field(default_factory=lambda: Path("data"))
    data_raw: Path = field(default_factory=lambda: Path("data/raw"))
    data_processed: Path = field(default_factory=lambda: Path("data/processed"))
    
    # Model checkpoints
    models_root: Path = field(default_factory=lambda: Path("models"))
    models_baseline: Path = field(default_factory=lambda: Path("models/baseline"))
    models_dpo: Path = field(default_factory=lambda: Path("models/dpo"))
    models_ppo: Path = field(default_factory=lambda: Path("models/ppo"))
    models_grpo: Path = field(default_factory=lambda: Path("models/grpo"))
    models_reward: Path = field(default_factory=lambda: Path("models/reward_model"))
    
    # Training runs and logs
    runs_root: Path = field(default_factory=lambda: Path("runs"))
    
    # Evaluation outputs
    eval_root: Path = field(default_factory=lambda: Path("eval"))
    eval_testset: Path = field(default_factory=lambda: Path("eval/testset_50.jsonl"))
    
    # Notebooks and analysis
    notebooks_root: Path = field(default_factory=lambda: Path("notebooks"))
    
    def create_directories(self):
        """Create all necessary directories"""
        dirs = [
            self.data_raw, self.data_processed,
            self.models_baseline, self.models_dpo, self.models_ppo, 
            self.models_grpo, self.models_reward,
            self.runs_root, self.eval_root, self.notebooks_root
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ARGUMENT PARSER FUNCTIONS
# =============================================================================

def add_base_args(parser: argparse.ArgumentParser):
    """Add base arguments common to all training scripts"""
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='HuggingFaceTB/SmolLM2-135M-Instruct',
                        help='Base model name or path')
    parser.add_argument('--load_in_8bit', action='store_true', default=True,
                        help='Load model in 8-bit quantization')
    parser.add_argument('--load_in_4bit', action='store_true', default=False,
                        help='Load model in 4-bit quantization')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    
    # LoRA arguments
    parser.add_argument('--use_lora', action='store_true', default=True,
                        help='Use LoRA adapters')
    parser.add_argument('--lora_r', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--lr', '--learning_rate', type=float, default=5e-5,
                        dest='learning_rate', help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Number of warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    
    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='adamw_torch',
                        choices=['adamw_torch', 'adamw_hf', 'sgd', 'adam'],
                        help='Optimizer type')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine',
                        choices=['linear', 'cosine', 'constant', 'polynomial'],
                        help='Learning rate scheduler type')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='HuggingFaceH5/orca_dpo_pairs',
                        help='Dataset name or path')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing processed data')
    
    # Logging and saving
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='Log every N steps')
    parser.add_argument('--save_steps', type=int, default=100,
                        help='Save checkpoint every N steps')
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='Evaluate every N steps')
    parser.add_argument('--save_total_limit', type=int, default=3,
                        help='Maximum number of checkpoints to keep')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Hardware
    parser.add_argument('--mixed_precision', type=str, default='fp16',
                        choices=['no', 'fp16', 'bf16'],
                        help='Mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Misc
    parser.add_argument('--output_dir', type=str, default='runs',
                        help='Output directory for runs')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Name for this run (auto-generated if None)')


def add_reward_model_args(parser: argparse.ArgumentParser):
    """Add reward model specific arguments"""
    parser.add_argument('--num_labels', type=int, default=1,
                        help='Number of output labels (1 for scalar reward)')
    parser.add_argument('--pairwise_training', action='store_true', default=True,
                        help='Use pairwise training')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Early stopping patience')


def add_dpo_args(parser: argparse.ArgumentParser):
    """Add DPO specific arguments"""
    parser.add_argument('--method', type=str, default='DPO',
                        help='Training method (for logging)')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='DPO beta parameter (temperature)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing')
    parser.add_argument('--loss_type', type=str, default='sigmoid',
                        choices=['sigmoid', 'hinge', 'ipo'],
                        help='DPO loss type')
    parser.add_argument('--reference_model', type=str, default=None,
                        help='Path to reference model (if None, uses frozen copy)')


def add_ppo_args(parser: argparse.ArgumentParser):
    """Add PPO specific arguments"""
    parser.add_argument('--method', type=str, default='PPO',
                        help='Training method (PPO-sparse or PPO-dense)')
    parser.add_argument('--reward_model_path', type=str, required=True,
                        help='Path to trained reward model')
    parser.add_argument('--reward_mode', type=str, default='sparse',
                        choices=['sparse', 'dense'],
                        help='Reward mode: sparse (final) or dense (token-level)')
    parser.add_argument('--kl_coef', '--init_kl_coef', type=float, default=0.05,
                        dest='kl_coef', help='KL divergence coefficient')
    parser.add_argument('--target_kl', type=float, default=0.1,
                        help='Target KL divergence')
    parser.add_argument('--clip_range', type=float, default=0.2,
                        help='PPO clipping parameter')
    parser.add_argument('--ppo_epochs', type=int, default=4,
                        help='Number of PPO epochs per batch')
    parser.add_argument('--mini_batch_size', type=int, default=2,
                        help='Mini batch size for PPO updates')
    parser.add_argument('--vf_coef', type=float, default=0.1,
                        help='Value function loss coefficient')
    parser.add_argument('--ent_coef', type=float, default=0.01,
                        help='Entropy coefficient')
    parser.add_argument('--max_new_tokens', type=int, default=128,
                        help='Maximum tokens to generate during rollouts')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p (nucleus) sampling')


def add_grpo_args(parser: argparse.ArgumentParser):
    """Add GRPO specific arguments"""
    parser.add_argument('--method', type=str, default='GRPO',
                        help='Training method (for logging)')
    parser.add_argument('--reward_model_path', type=str, required=True,
                        help='Path to trained reward model')
    parser.add_argument('--group_size', type=int, default=8,
                        help='Number of candidate responses per prompt')
    parser.add_argument('--advantage_normalization', type=str, default='rank',
                        choices=['rank', 'zscore', 'minmax'],
                        help='Advantage normalization method')
    parser.add_argument('--use_baseline', action='store_true', default=True,
                        help='Use group mean as baseline')
    parser.add_argument('--max_new_tokens', type=int, default=128,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p (nucleus) sampling')


def add_eval_args(parser: argparse.ArgumentParser):
    """Add evaluation specific arguments"""
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint to evaluate')
    parser.add_argument('--reward_model_path', type=str, default=None,
                        help='Path to reward model (for computing rewards)')
    parser.add_argument('--reference_model_path', type=str, default=None,
                        help='Path to reference model (for computing KL)')
    parser.add_argument('--test_file', type=str, default='eval/testset_50.jsonl',
                        help='Path to test prompts')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save evaluation results')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples per prompt')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--compute_perplexity', action='store_true', default=True,
                        help='Compute perplexity on instruction-following data')


def parse_reward_model_args():
    """Parse arguments for reward model training"""
    parser = argparse.ArgumentParser(description='Train Reward Model')
    add_base_args(parser)
    add_reward_model_args(parser)
    args = parser.parse_args()
    return args


def parse_dpo_args():
    """Parse arguments for DPO training"""
    parser = argparse.ArgumentParser(description='Train DPO')
    add_base_args(parser)
    add_dpo_args(parser)
    args = parser.parse_args()
    return args


def parse_ppo_args():
    """Parse arguments for PPO training"""
    parser = argparse.ArgumentParser(description='Train PPO')
    add_base_args(parser)
    add_ppo_args(parser)
    args = parser.parse_args()
    return args


def parse_grpo_args():
    """Parse arguments for GRPO training"""
    parser = argparse.ArgumentParser(description='Train GRPO')
    add_base_args(parser)
    add_grpo_args(parser)
    args = parser.parse_args()
    return args


def parse_eval_args():
    """Parse arguments for evaluation"""
    parser = argparse.ArgumentParser(description='Evaluate Model')
    add_base_args(parser)
    add_eval_args(parser)
    args = parser.parse_args()
    return args


def args_to_config(args: argparse.Namespace) -> 'Config':
    """
    Convert argparse Namespace to Config object
    This allows seamless integration between CLI args and config dataclasses
    """
    config = Config()
    
    # Update base model config
    if hasattr(args, 'model_name'):
        config.base_model.model_name = args.model_name
    if hasattr(args, 'load_in_8bit'):
        config.base_model.load_in_8bit = args.load_in_8bit
    if hasattr(args, 'load_in_4bit'):
        config.base_model.load_in_4bit = args.load_in_4bit
    if hasattr(args, 'max_length'):
        config.base_model.max_length = args.max_length
    
    # Update LoRA config
    if hasattr(args, 'use_lora'):
        config.lora.use_lora = args.use_lora
    if hasattr(args, 'lora_r'):
        config.lora.lora_r = args.lora_r
    if hasattr(args, 'lora_alpha'):
        config.lora.lora_alpha = args.lora_alpha
    if hasattr(args, 'lora_dropout'):
        config.lora.lora_dropout = args.lora_dropout
    
    # Update data config
    if hasattr(args, 'dataset'):
        config.data.dataset_name = args.dataset
    if hasattr(args, 'seed'):
        config.data.seed = args.seed
    
    # Update training configs based on method
    if hasattr(args, 'method'):
        method = args.method.upper()
        
        if 'DPO' in method:
            if hasattr(args, 'learning_rate'):
                config.dpo.learning_rate = args.learning_rate
            if hasattr(args, 'batch_size'):
                config.dpo.batch_size = args.batch_size
            if hasattr(args, 'gradient_accumulation_steps'):
                config.dpo.gradient_accumulation_steps = args.gradient_accumulation_steps
            if hasattr(args, 'epochs'):
                config.dpo.num_epochs = args.epochs
            if hasattr(args, 'warmup_steps'):
                config.dpo.warmup_steps = args.warmup_steps
            if hasattr(args, 'weight_decay'):
                config.dpo.weight_decay = args.weight_decay
            if hasattr(args, 'max_grad_norm'):
                config.dpo.max_grad_norm = args.max_grad_norm
            if hasattr(args, 'beta'):
                config.dpo.beta = args.beta
            if hasattr(args, 'loss_type'):
                config.dpo.loss_type = args.loss_type
                
        elif 'PPO' in method:
            if hasattr(args, 'learning_rate'):
                config.ppo.learning_rate = args.learning_rate
            if hasattr(args, 'batch_size'):
                config.ppo.batch_size = args.batch_size
            if hasattr(args, 'gradient_accumulation_steps'):
                config.ppo.gradient_accumulation_steps = args.gradient_accumulation_steps
            if hasattr(args, 'epochs'):
                config.ppo.num_epochs = args.epochs
            if hasattr(args, 'warmup_steps'):
                config.ppo.warmup_steps = args.warmup_steps
            if hasattr(args, 'weight_decay'):
                config.ppo.weight_decay = args.weight_decay
            if hasattr(args, 'max_grad_norm'):
                config.ppo.max_grad_norm = args.max_grad_norm
            if hasattr(args, 'reward_model_path'):
                config.ppo.reward_model_path = args.reward_model_path
            if hasattr(args, 'reward_mode'):
                config.ppo.reward_mode = args.reward_mode
            if hasattr(args, 'kl_coef'):
                config.ppo.init_kl_coef = args.kl_coef
            if hasattr(args, 'target_kl'):
                config.ppo.target_kl = args.target_kl
            if hasattr(args, 'clip_range'):
                config.ppo.clip_range = args.clip_range
            if hasattr(args, 'ppo_epochs'):
                config.ppo.ppo_epochs = args.ppo_epochs
            if hasattr(args, 'mini_batch_size'):
                config.ppo.mini_batch_size = args.mini_batch_size
            if hasattr(args, 'vf_coef'):
                config.ppo.vf_coef = args.vf_coef
            if hasattr(args, 'ent_coef'):
                config.ppo.ent_coef = args.ent_coef
            if hasattr(args, 'max_new_tokens'):
                config.ppo.max_new_tokens = args.max_new_tokens
            if hasattr(args, 'temperature'):
                config.ppo.temperature = args.temperature
            if hasattr(args, 'top_k'):
                config.ppo.top_k = args.top_k
            if hasattr(args, 'top_p'):
                config.ppo.top_p = args.top_p
                
        elif 'GRPO' in method:
            if hasattr(args, 'learning_rate'):
                config.grpo.learning_rate = args.learning_rate
            if hasattr(args, 'batch_size'):
                config.grpo.batch_size = args.batch_size
            if hasattr(args, 'gradient_accumulation_steps'):
                config.grpo.gradient_accumulation_steps = args.gradient_accumulation_steps
            if hasattr(args, 'epochs'):
                config.grpo.num_epochs = args.epochs
            if hasattr(args, 'warmup_steps'):
                config.grpo.warmup_steps = args.warmup_steps
            if hasattr(args, 'weight_decay'):
                config.grpo.weight_decay = args.weight_decay
            if hasattr(args, 'max_grad_norm'):
                config.grpo.max_grad_norm = args.max_grad_norm
            if hasattr(args, 'reward_model_path'):
                config.grpo.reward_model_path = args.reward_model_path
            if hasattr(args, 'group_size'):
                config.grpo.group_size = args.group_size
            if hasattr(args, 'advantage_normalization'):
                config.grpo.advantage_normalization = args.advantage_normalization
            if hasattr(args, 'use_baseline'):
                config.grpo.use_baseline = args.use_baseline
            if hasattr(args, 'max_new_tokens'):
                config.grpo.max_new_tokens = args.max_new_tokens
            if hasattr(args, 'temperature'):
                config.grpo.temperature = args.temperature
            if hasattr(args, 'top_k'):
                config.grpo.top_k = args.top_k
            if hasattr(args, 'top_p'):
                config.grpo.top_p = args.top_p
    
    # For reward model training
    if hasattr(args, 'num_labels'):
        if hasattr(args, 'learning_rate'):
            config.reward_model.learning_rate = args.learning_rate
        if hasattr(args, 'batch_size'):
            config.reward_model.batch_size = args.batch_size
        if hasattr(args, 'gradient_accumulation_steps'):
            config.reward_model.gradient_accumulation_steps = args.gradient_accumulation_steps
        if hasattr(args, 'epochs'):
            config.reward_model.num_epochs = args.epochs
        if hasattr(args, 'warmup_steps'):
            config.reward_model.warmup_steps = args.warmup_steps
        if hasattr(args, 'weight_decay'):
            config.reward_model.weight_decay = args.weight_decay
        if hasattr(args, 'max_grad_norm'):
            config.reward_model.max_grad_norm = args.max_grad_norm
    
    # Update logging config
    if hasattr(args, 'logging_steps'):
        config.logging.log_metrics_every_n_steps = args.logging_steps
    
    # Update hardware config
    if hasattr(args, 'mixed_precision'):
        config.hardware.mixed_precision = args.mixed_precision
    if hasattr(args, 'num_workers'):
        config.hardware.num_workers = args.num_workers
    
    return config


def save_args_to_json(args: argparse.Namespace, output_path: str):
    """Save command-line arguments to JSON file"""
    args_dict = vars(args)
    # Convert Path objects to strings
    for key, value in args_dict.items():
        if isinstance(value, Path):
            args_dict[key] = str(value)
    
    with open(output_path, 'w') as f:
        json.dump(args_dict, f, indent=2)


@dataclass
class BaseModelConfig:
    """Configuration for the base model and tokenizer"""
    
    # Model identifier
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    
    # Quantization settings
    load_in_8bit: bool = True  # Use 8-bit quantization to save memory
    load_in_4bit: bool = False  # Alternative: 4-bit quantization
    
    # Device and precision
    device_map: str = "auto"  # Automatic device mapping
    torch_dtype: str = "auto"  # Auto-detect best dtype
    
    # Tokenizer settings
    tokenizer_name: Optional[str] = None  # If None, use model_name
    max_length: int = 512  # Maximum sequence length
    padding_side: str = "left"  # For generation
    truncation_side: str = "right"
    
    # Trust remote code (if needed)
    trust_remote_code: bool = False


# =============================================================================
# LORA / PEFT CONFIGURATION
# =============================================================================

@dataclass
class LoRAConfig:
    """LoRA adapter configuration for parameter-efficient fine-tuning"""
    
    # Enable LoRA
    use_lora: bool = True
    
    # LoRA hyperparameters
    lora_r: int = 8  # Rank - ablation sweep: [4, 8, 16]
    lora_alpha: int = 16  # Scaling factor (typically 2*r)
    lora_dropout: float = 0.05
    
    # Target modules (model-specific, adjust for SmolLM architecture)
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention projections
        "gate_proj", "up_proj", "down_proj"  # FFN projections
    ])
    
    # Task type
    task_type: str = "CAUSAL_LM"
    
    # Bias training
    bias: str = "none"  # Options: "none", "all", "lora_only"
    
    # Additional LoRA settings
    modules_to_save: Optional[List[str]] = None  # Additional modules to train fully


# =============================================================================
# DATA PREPARATION
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for data preparation and processing"""
    
    # Dataset source
    dataset_name: str = "HuggingFaceH5/orca_dpo_pairs"  # Or path to local file
    dataset_config: Optional[str] = None
    
    # Splits
    train_split: str = "train"
    val_split: Optional[str] = None  # If None, will split from train
    test_split: Optional[str] = None
    
    # Split ratios (if creating splits manually)
    train_ratio: float = 0.85
    val_ratio: float = 0.10
    test_ratio: float = 0.05
    
    # Formatting
    prompt_template: str = "<|user|>\n{prompt}\n<|assistant|>\n"
    response_template: str = "{response}"
    
    # Instruction-following subset for perplexity evaluation
    instr_following_size: int = 5000  # Number of examples
    
    # Data filtering
    min_prompt_length: int = 10  # Minimum prompt length in characters
    max_prompt_length: int = 2048  # Maximum prompt length in tokens
    min_response_length: int = 5
    max_response_length: int = 1024
    
    # Seed for reproducibility
    seed: int = 42


# =============================================================================
# REWARD MODEL TRAINING
# =============================================================================

@dataclass
class RewardModelConfig:
    """Configuration for reward model training"""
    
    # Model architecture
    base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    num_labels: int = 1  # Scalar reward output
    
    # Training mode
    pairwise_training: bool = True  # Train on pairs (yW vs yL)
    
    # Training hyperparameters
    num_epochs: int = 3
    learning_rate: float = 5e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Optimizer
    optimizer: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"
    
    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Logging
    logging_steps: int = 10
    
    # Capacity ablation variants
    capacity_variants: List[str] = field(default_factory=lambda: ["small", "base", "large"])
    
    # Seeds for multi-seed training
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])


# =============================================================================
# DPO TRAINING
# =============================================================================

@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization"""
    
    # Training hyperparameters
    num_epochs: int = 3
    learning_rate: float = 5e-5  # Sweep: [1e-4, 5e-5, 1e-5]
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    
    # DPO-specific parameters
    beta: float = 0.1  # Temperature parameter for DPO loss
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"  # Options: "sigmoid", "hinge", "ipo"
    
    # Reference model
    use_frozen_reference: bool = True  # Use frozen copy of base model
    
    # Optimizer settings
    optimizer: str = "adamw_torch"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"
    
    # Evaluation and saving
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    
    # Logging
    logging_steps: int = 10
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    
    # Ablations
    lora_variants: List[int] = field(default_factory=lambda: [4, 8, 16])  # LoRA rank sweep
    lr_variants: List[float] = field(default_factory=lambda: [1e-4, 5e-5, 1e-5])
    full_finetune: bool = False  # Also run without LoRA
    
    # Seeds for multi-seed experiments
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])


# =============================================================================
# PPO TRAINING
# =============================================================================

@dataclass
class PPOConfig:
    """Configuration for Proximal Policy Optimization"""
    
    # Training hyperparameters
    num_epochs: int = 3
    learning_rate: float = 1e-5
    batch_size: int = 8
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    
    # PPO-specific parameters
    ppo_epochs: int = 4  # Number of PPO update epochs per batch
    clip_range: float = 0.2  # Clipping parameter for PPO
    clip_range_vf: Optional[float] = None  # Value function clipping (if None, same as clip_range)
    
    # KL divergence penalty (critical for controlling drift)
    kl_penalty: str = "kl"  # Options: "kl", "abs", "mse", "full"
    target_kl: Optional[float] = 0.1  # Target KL divergence
    init_kl_coef: float = 0.05  # Initial KL coefficient - SWEEP: [0.01, 0.05, 0.1]
    adap_kl_ctrl: bool = True  # Adaptive KL control
    
    # Entropy bonus (exploration)
    ent_coef: float = 0.01
    
    # Value function
    vf_coef: float = 0.1  # Value function loss coefficient
    
    # Reward settings
    reward_model_path: Optional[str] = None  # Path to trained reward model
    reward_mode: str = "sparse"  # Options: "sparse", "dense"
    
    # For dense rewards
    token_level_rewards: bool = False  # Compute rewards at token level
    intermediate_reward_weight: float = 0.1  # Weight for intermediate rewards
    
    # Generation settings for rollouts
    max_new_tokens: int = 128
    temperature: float = 0.7  # Sampling temperature - SWEEP: [0.2, 0.7, 1.0]
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    
    # Optimization
    optimizer: str = "adamw_torch"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 10
    
    # Logging and checkpointing
    logging_steps: int = 1
    save_freq: int = 100
    eval_freq: int = 100
    
    # Ablations
    kl_coef_variants: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])
    no_kl_ablation: bool = True  # Run PPO without KL penalty
    large_kl_ablation: bool = True  # Run PPO with large KL (0.5)
    temperature_variants: List[float] = field(default_factory=lambda: [0.2, 0.7, 1.0])
    
    # Reference policy variants (for ablation)
    reference_policy_variants: List[str] = field(default_factory=lambda: ["sft", "slightly_finetuned"])
    
    # Value model configuration
    value_model_lora: bool = True
    value_model_lr: float = 1e-5
    
    # Seeds
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])


# =============================================================================
# GRPO TRAINING
# =============================================================================

@dataclass
class GRPOConfig:
    """Configuration for Group Relative Policy Optimization"""
    
    # Training hyperparameters
    num_epochs: int = 3
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    
    # GRPO-specific parameters
    group_size: int = 8  # Number of candidates per prompt - SWEEP: [4, 8, 16]
    min_group_size: int = 4  # Minimum required
    
    # Advantage normalization
    advantage_normalization: str = "rank"  # Options: "rank", "zscore", "minmax"
    use_baseline: bool = True  # Subtract group mean for advantages
    
    # Reward model
    reward_model_path: Optional[str] = None  # Use same as PPO
    
    # Generation settings
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    
    # Optimization
    optimizer: str = "adamw_torch"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 10
    
    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"
    
    # Logging and checkpointing
    logging_steps: int = 1
    save_freq: int = 100
    eval_freq: int = 100
    
    # Ablations
    group_size_variants: List[int] = field(default_factory=lambda: [4, 8, 16])
    normalization_variants: List[str] = field(default_factory=lambda: ["rank", "zscore", "minmax"])
    with_without_normalization: bool = True  # Test GRPO with/without normalization
    
    # Seeds
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])


# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

@dataclass
class EvalConfig:
    """Configuration for evaluation pipeline"""
    
    # Test set
    num_test_prompts: int = 50
    
    # Test set categories (distribution)
    category_distribution: Dict[str, int] = field(default_factory=lambda: {
        "factual_short": 10,  # Factual short-answer questions
        "open_ended": 10,  # Open-ended explanatory prompts
        "instruction_following": 10,  # Explicit instruction-following
        "verbosity_hack": 10,  # Prompts encouraging length
        "adversarial": 10  # Reward-hacking adversarial prompts
    })
    
    # Generation settings for evaluation
    generation_configs: List[Dict] = field(default_factory=lambda: [
        {"name": "greedy", "do_sample": False, "num_return_sequences": 1},
        {"name": "top_k_5", "do_sample": True, "top_k": 5, "num_return_sequences": 1},
        {"name": "temp_0.2", "do_sample": True, "temperature": 0.2, "num_return_sequences": 1},
        {"name": "temp_0.7", "do_sample": True, "temperature": 0.7, "num_return_sequences": 1},
        {"name": "temp_1.0", "do_sample": True, "temperature": 1.0, "num_return_sequences": 1},
    ])
    
    # Metrics to compute
    compute_reward_scores: bool = True
    compute_perplexity: bool = True
    compute_kl_divergence: bool = True
    compute_length_stats: bool = True
    compute_quality_proxy: bool = True
    compute_compliance_rate: bool = True
    
    # Perplexity computation
    perplexity_dataset_size: int = 1000  # Subset for perplexity
    
    # Quality proxy settings
    quality_metrics: List[str] = field(default_factory=lambda: [
        "rouge_l", "bleu", "exact_match", "perplexity_delta"
    ])
    
    # Compliance testing
    length_constraint_prompts: int = 10  # Number of prompts with explicit length limits


# =============================================================================
# PERTURBATION AND REWARD HACKING TESTS
# =============================================================================

@dataclass
class PerturbationConfig:
    """Configuration for reward hacking and perturbation experiments"""
    
    # Surface-form perturbations
    perturbation_types: List[str] = field(default_factory=lambda: [
        "filler_phrases",  # Add "To be honest,", "Actually,", etc.
        "sentence_reorder",  # Reorder sentences
        "synonym_replacement",  # Replace words with synonyms
        "alignment_keywords",  # Inject "As an AI...", "I cannot..."
        "formatting_changes"  # Add bullet points, numbering, etc.
    ])
    
    # Filler phrases to test
    filler_phrases: List[str] = field(default_factory=lambda: [
        "To be honest, ",
        "Actually, ",
        "In my opinion, ",
        "I think ",
        "It's important to note that ",
        "As a matter of fact, "
    ])
    
    # Alignment keywords
    alignment_keywords: List[str] = field(default_factory=lambda: [
        "As an AI language model, ",
        "I cannot provide ",
        "I'm designed to ",
        "This is likely safe ",
        "I apologize, but "
    ])
    
    # Adversarial prompt types
    adversarial_types: List[str] = field(default_factory=lambda: [
        "impossible_task",  # Contradictory or impossible requests
        "vague_safety",  # Vague safety-themed questions
        "instruction_loop",  # Circular or confusing instructions
        "keyword_stuffing",  # Stuffed with reward-triggering keywords
        "contradictory"  # Contradictory requirements
    ])
    
    # Thresholds for flagging hacking
    reward_delta_threshold: float = 0.2  # Significant reward change
    quality_drop_threshold: float = 0.3  # Significant quality drop
    
    # Number of perturbations per example
    num_perturbations_per_example: int = 5


# =============================================================================
# ALIGNMENT TAX / CATASTROPHIC FORGETTING
# =============================================================================

@dataclass
class AlignmentTaxConfig:
    """Configuration for measuring alignment tax and catastrophic forgetting"""
    
    # KL divergence computation
    compute_kl_per_sample: bool = True
    kl_samples: int = 100  # Number of samples for KL estimation
    
    # Perplexity on instruction-following data
    perplexity_eval_size: int = 5000  # Size of held-out SFT data
    
    # Task-specific microbenchmarks
    microbenchmark_tasks: List[str] = field(default_factory=lambda: [
        "instruction_following",
        "factual_qa",
        "summarization",
        "reasoning"
    ])
    
    # Comparison baselines
    compare_to_sft: bool = True


# =============================================================================
# VERBOSITY BIAS ANALYSIS
# =============================================================================

@dataclass
class VerbosityConfig:
    """Configuration for verbosity bias analysis"""
    
    # Length statistics to compute
    length_metrics: List[str] = field(default_factory=lambda: [
        "mean", "median", "std", "min", "max", "skewness", "kurtosis"
    ])
    
    # Stratification
    stratify_by_category: bool = True
    
    # Compliance testing
    test_length_constraints: bool = True
    length_constraints: List[Tuple[int, str]] = field(default_factory=lambda: [
        (20, "in 20 words or less"),
        (50, "in 50 words or less"),
        (100, "in 100 words or less")
    ])
    
    # Rambling detection
    rambling_threshold_std: float = 3.0  # Flag outputs > mean + 3*std


# =============================================================================
# PLOTTING AND VISUALIZATION
# =============================================================================

@dataclass
class PlottingConfig:
    """Configuration for generating plots and visualizations"""
    
    # Plot formats
    save_formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    dpi: int = 300
    
    # Plot types to generate
    plot_types: List[str] = field(default_factory=lambda: [
        "training_curves",  # Loss, reward, KL vs steps
        "kl_perplexity_scatter",  # KL vs perplexity
        "reward_quality_scatter",  # Reward vs quality
        "reward_distributions",  # Violin/box plots
        "length_distributions",  # Histograms and boxplots
        "compliance_bars",  # Compliance rate bar chart
        "perturbation_sensitivity",  # Delta reward under perturbations
        "ablation_heatmaps",  # Heatmaps for ablations
        "group_advantages",  # GRPO advantage distributions
        "confidence_intervals"  # Error bars from multi-seed runs
    ])
    
    # Style
    style: str = "seaborn-v0_8-darkgrid"
    color_palette: str = "Set2"


# =============================================================================
# LOGGING AND REPRODUCIBILITY
# =============================================================================

@dataclass
class LoggingConfig:
    """Configuration for logging and reproducibility"""
    
    # Logging format
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level: str = "INFO"
    
    # Metrics logging
    log_metrics_every_n_steps: int = 10
    save_metrics_csv: bool = True
    save_metrics_jsonl: bool = True
    
    # Checkpoint saving
    save_checkpoints_every_n_epochs: int = 1
    save_final_checkpoint: bool = True
    save_optimizer_state: bool = True
    save_tokenizer: bool = True
    
    # Reproducibility
    set_seed_everywhere: bool = True
    deterministic_algorithms: bool = True
    
    # Run tracking
    track_in_runs_index: bool = True  # Maintain runs/index.csv
    
    # TensorBoard / Weights & Biases
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: Optional[str] = None


# =============================================================================
# HARDWARE AND PERFORMANCE
# =============================================================================

@dataclass
class HardwareConfig:
    """Configuration for hardware and performance settings"""
    
    # GPU settings
    num_gpus: int = 1
    mixed_precision: str = "fp16"  # Options: "no", "fp16", "bf16"
    
    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Gradient checkpointing
    gradient_checkpointing: bool = True
    
    # Memory optimization
    offload_to_cpu: bool = False
    empty_cache_steps: int = 10


# =============================================================================
# STATISTICAL TESTING
# =============================================================================

@dataclass
class StatisticalConfig:
    """Configuration for statistical tests and confidence intervals"""
    
    # Multi-seed analysis
    num_seeds: int = 3
    confidence_level: float = 0.95
    
    # Statistical tests
    run_paired_tests: bool = True  # Paired t-test or Wilcoxon
    test_metrics: List[str] = field(default_factory=lambda: [
        "reward", "perplexity", "compliance_rate", "kl_divergence"
    ])
    
    # Significance threshold
    alpha: float = 0.05


# =============================================================================
# MAIN CONFIG CLASS (AGGREGATES ALL CONFIGS)
# =============================================================================

@dataclass
class Config:
    """Master configuration class aggregating all sub-configs"""
    
    paths: PathConfig = field(default_factory=PathConfig)
    base_model: BaseModelConfig = field(default_factory=BaseModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    alignment_tax: AlignmentTaxConfig = field(default_factory=AlignmentTaxConfig)
    verbosity: VerbosityConfig = field(default_factory=VerbosityConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    statistical: StatisticalConfig = field(default_factory=StatisticalConfig)
    
    def __post_init__(self):
        """Create directories after initialization"""
        self.paths.create_directories()
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        import json
        from dataclasses import asdict
        
        config_dict = asdict(self)
        # Convert Path objects to strings
        config_dict = self._convert_paths_to_str(config_dict)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _convert_paths_to_str(self, obj):
        """Recursively convert Path objects to strings"""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_paths_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_paths_to_str(item) for item in obj]
        else:
            return obj
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file"""
        import json
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct Config object from dict
        # This is simplified; you may need more complex reconstruction logic
        return cls(**config_dict)



# =============================================================================
# BASE MODEL AND TOKENIZER
# =============================================================================


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_default_config() -> Config:
    """Get default configuration instance"""
    return Config()


def get_config_from_args(args: argparse.Namespace) -> Config:
    """
    Create Config object from command-line arguments
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Config object with values from args
    """
    return args_to_config(args)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Example 1: Get default config
    print("=" * 80)
    print("Example 1: Default Config")
    print("=" * 80)
    config = get_default_config()
    print(f"Base model: {config.base_model.model_name}")
    print(f"LoRA rank: {config.lora.lora_r}")
    print(f"DPO learning rate: {config.dpo.learning_rate}")
    print(f"PPO KL coefficient: {config.ppo.init_kl_coef}")
    print(f"GRPO group size: {config.grpo.group_size}")
    
    # Example 2: Demonstrate argparse usage
    print("\n" + "=" * 80)
    print("Example 2: Command-line Usage Examples")
    print("=" * 80)
    print("\n# Train DPO:")
    print("python train_dpo.py \\")
    print("  --method DPO \\")
    print("  --batch_size 8 \\")
    print("  --lr 1e-4 \\")
    print("  --epochs 3 \\")
    print("  --lora_r 8 \\")
    print("  --beta 0.1 \\")
    print("  --seed 42 \\")
    print("  --save_dir ./checkpoints/dpo")
    
    print("\n# Train PPO (sparse reward):")
    print("python train_ppo.py \\")
    print("  --method PPO \\")
    print("  --reward_model_path ./models/reward_model/best \\")
    print("  --reward_mode sparse \\")
    print("  --batch_size 8 \\")
    print("  --lr 1e-5 \\")
    print("  --kl_coef 0.05 \\")
    print("  --epochs 3 \\")
    print("  --seed 42 \\")
    print("  --save_dir ./checkpoints/ppo_sparse")
    
    print("\n# Train GRPO:")
    print("python train_grpo.py \\")
    print("  --method GRPO \\")
    print("  --reward_model_path ./models/reward_model/best \\")
    print("  --group_size 8 \\")
    print("  --advantage_normalization rank \\")
    print("  --batch_size 4 \\")
    print("  --lr 1e-5 \\")
    print("  --epochs 3 \\")
    print("  --seed 42 \\")
    print("  --save_dir ./checkpoints/grpo")
    
    print("\n# Train Reward Model:")
    print("python train_reward_model.py \\")
    print("  --batch_size 16 \\")
    print("  --lr 5e-5 \\")
    print("  --epochs 3 \\")
    print("  --seed 42 \\")
    print("  --save_dir ./models/reward_model")
    
    print("\n# Evaluate Model:")
    print("python eval_generate.py \\")
    print("  --model_path ./checkpoints/dpo/best \\")
    print("  --reward_model_path ./models/reward_model/best \\")
    print("  --test_file ./eval/testset_50.jsonl \\")
    print("  --temperature 0.7 \\")
    print("  --output_file ./eval/dpo_results.csv")
    
    # Save default config
    print("\n" + "=" * 80)
    config.save("default_config.json")
    print("Default configuration saved to default_config.json")
    print("=" * 80)