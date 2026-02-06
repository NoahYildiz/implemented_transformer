"""
Configuration for Transformer training.
All hyperparameters in one place - no defaults scattered around.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Complete configuration for Transformer training."""
    
    # =========================================================================
    # Model Architecture (from "Attention is All You Need")
    # =========================================================================
    d_model: int = 512          # Model dimension
    n_heads: int = 8            # Number of attention heads
    d_ff: int = 2048            # Feed-forward hidden dimension
    n_layers: int = 6           # Number of encoder/decoder layers
    dropout: float = 0.1        # Dropout probability
    max_len: int = 192          # Maximum sequence length
    use_rope: bool = True       # Whether to use Rotary Positional Embeddings
    use_adamw: bool = True      # Whether to use AdamW (vs plain Adam) for optimizer
    
    # =========================================================================
    # Vocabulary & Special Tokens
    # =========================================================================
    vocab_size: int = 32000     # Vocabulary size
    pad_id: int = 1             # [PAD] token ID
    bos_id: int = 2             # [BOS] token ID
    eos_id: int = 3             # [EOS] token ID
    unk_id: int = 4             # [UNK] token ID
    
    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    batch_size: int = 64        # Batch size
    epochs: int = 5            # Number of epochs
    warmup_steps: int = 4000    # LR warmup steps
    label_smoothing: float = 0.1  # Label smoothing
    max_grad_norm: float = 1.0  # Gradient clipping
    
    # Adam / AdamW optimizer (paper: β1=0.9, β2=0.98, ε=10^-9)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1e-9
    adamw_weight_decay: float = 0.01  # Only used when use_adamw=True
    
    # =========================================================================
    # Data
    # =========================================================================
    train_samples: int = None   # None = use all
    val_samples: int = 3000     # Validation samples
    src_lang: str = "de"        # Source language
    tgt_lang: str = "en"        # Target language
    cache_dir: str = "/data/cat/ws/tosa098h-transformer-ws/noah/cache_huggingface"
    
    # =========================================================================
    # Paths
    # =========================================================================
    output_dir: str = "/data/cat/ws/tosa098h-transformer-ws/noah/checkpoints"
    log_dir: str = "./logs"
    tokenizer_path: str = "./tokenizer_data"
    
    # =========================================================================
    # Logging & Validation
    # =========================================================================
    validate_every: int = 5000  # Steps between validation
    save_every: int = 20000     # Steps between checkpoints
    log_every: int = 100        # Steps between logging
    bleu_samples: int = 150      # Samples for BLEU calculation
    
    # WandB
    use_wandb: bool = True
    wandb_project: str = "transformer2"
    wandb_run_name: str = "rope_bfloat16"
    
    # =========================================================================
    # Hardware
    # =========================================================================
    use_amp: bool = True        # Mixed precision training
    device: str = "cuda"        # Device (cuda/cpu)
    
    # =========================================================================
    # Debug
    # =========================================================================
    debug: bool = True          # Enable debug prints
    
    def __post_init__(self):
        """Create directories if needed."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def print_config(self):
        """Print configuration."""
        print("\n" + "=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        for field_name, field_value in self.__dict__.items():
            print(f"  {field_name}: {field_value}")
        print("=" * 60 + "\n")


# Default configuration
def get_config() -> Config:
    """Get default configuration."""
    return Config()
