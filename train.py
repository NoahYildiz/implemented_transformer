"""
Training script for Transformer with WandB logging and debug output.

Clean implementation using config.py for all hyperparameters.
"""
import os
import math
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.amp import autocast  # No GradScaler needed for BFloat16
from tqdm import tqdm

from config import Config, get_config
from evaluate import validate
from model import Transformer
from tokenizer import BPETokenizer
from dataset import load_wmt17, TranslationDataset, Collator, create_dataloaders, debug_batch

# Setup Logger
logger = logging.getLogger(__name__)

def setup_logging(log_dir: str):
    """Setup logging to file."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    # We do not add StreamHandler to avoid conflict with tqdm
    
    return log_file

def log_info(msg: str):
    """Log message to file and print to console (tqdm-safe)."""
    logger.info(msg)
    tqdm.write(msg)

class TransformerLRScheduler:
    """
    Learning rate schedule from "Attention is All You Need":
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def _get_lr(self):
        step = max(self.step_num, 1)
        return (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))


def debug_print(config: Config, msg: str):
    """Print debug message if debug mode is enabled."""
    if config.debug:
        log_info(f"[DEBUG] {msg}")


def train_epoch(
    model: Transformer,
    dataloader,
    optimizer,
    scheduler: TransformerLRScheduler,
    criterion: nn.Module,
    config: Config,
    epoch: int,
    global_step: int,
    use_amp: bool = False,
    wandb_run=None,
    val_loader=None,
    tokenizer=None,
) -> tuple:
    """Train for one epoch."""
    model.train()
    device = torch.device(config.device)
    total_loss = 0
    total_tokens = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        labels = batch['labels'].to(device)
        src_mask = batch['src_mask'].to(device)
        tgt_mask = batch['tgt_mask'].to(device)
        
        # Debug first batch
        if batch_idx == 0 and epoch == 1 and config.debug:
            log_info("\n" + "=" * 60)
            log_info("FIRST BATCH DEBUG")
            log_info("=" * 60)
            log_info(f"src shape: {src.shape}")
            log_info(f"tgt shape: {tgt.shape}")
            log_info(f"labels shape: {labels.shape}")
            log_info(f"src_mask shape: {src_mask.shape}")
            log_info(f"tgt_mask shape: {tgt_mask.shape}")
            log_info(f"\nFirst example:")
            log_info(f"  src[:10]: {src[0, :10].tolist()}")
            log_info(f"  tgt[:10]: {tgt[0, :10].tolist()}")
            log_info(f"  labels[:10]: {labels[0, :10].tolist()}")
            log_info(f"\nMask info:")
            log_info(f"  src_mask valid: {src_mask[0].sum().item()}")
            log_info(f"  tgt_mask shape check: {tgt_mask[0, 0].shape}")
            log_info("=" * 60 + "\n")
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision (BFloat16)
        with autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            logits = model(src, tgt, src_mask, tgt_mask)
            
            # Debug logits on first batch
            if batch_idx == 0 and epoch == 1 and config.debug:
                log_info(f"\n[DEBUG] Logits shape: {logits.shape}")
                log_info(f"[DEBUG] Logits min/max: {logits.min().item():.4f} / {logits.max().item():.4f}")
                log_info(f"[DEBUG] Logits mean/std: {logits.mean().item():.4f} / {logits.std().item():.4f}")
            
            # Reshape for loss: (batch * seq, vocab) vs (batch * seq)
            loss = criterion(logits.float().view(-1, logits.size(-1)), labels.view(-1))
            
            if batch_idx == 0 and epoch == 1 and config.debug:
                log_info(f"[DEBUG] Loss: {loss.item():.4f}")
                log_info(f"[DEBUG] Expected initial loss ~ln(vocab_size) = {math.log(logits.size(-1)):.4f}\n")
            
            # Debug: Check for loss explosion in first 10 batches
            if batch_idx < 10 and epoch == 1 and config.debug:
                log_info(f"[DEBUG] Batch {batch_idx}: loss={loss.item():.4f}, logits_max={logits.max().item():.2f}, logits_min={logits.min().item():.2f}")
        
        # Backward pass - no scaler needed for BFloat16
        loss.backward()
        
        # Debug gradients
        if batch_idx < 10 and epoch == 1 and config.debug:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            log_info(f"[DEBUG] Batch {batch_idx} Grad Norm: {total_norm:.4f}")
            if math.isnan(total_norm) or math.isinf(total_norm):
                log_info(f"[DEBUG] 🚨 Infinite/NaN Gradient detected!")
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
        optimizer.step()
        
        lr = scheduler.step()
        
        # Statistics (count only non-padding tokens)
        num_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        global_step += 1
        
        # Update progress bar
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}',
            'lr': f'{lr:.2e}',
            'step': global_step
        })
        
        # Log to WandB
        if wandb_run is not None and global_step % config.log_every == 0:
            import wandb
            wandb.log({
                'train/loss': loss.item(),
                'train/lr': lr,
                'train/step': global_step
            }, step=global_step)
        
        # Mid-epoch validation
        if val_loader is not None and tokenizer is not None and global_step % config.validate_every == 0:
            pbar.write(f"\n[Step {global_step}] Running Validation...")
            model.eval()
            val_loss, bleu = validate(
                model, val_loader, criterion, tokenizer, device, config,
                max_batches=config.val_samples // config.batch_size + 1
            )
            log_info(f"[Step {global_step}] Val Loss: {val_loss:.4f}, BLEU: {bleu:.2f}")
            
            # Show sample translations
            generate_samples(model, val_loader, tokenizer, config, n_samples=3)
            
            if wandb_run is not None:
                import wandb
                wandb.log({
                    'val/loss': val_loss,
                    'val/bleu': bleu,
                }, step=global_step)
            
            model.train()  # Back to training mode
    
    return total_loss / total_tokens if total_tokens > 0 else 0, global_step


@torch.no_grad()
def generate_samples(model: Transformer, dataloader, tokenizer, config: Config, n_samples: int = 3):
    """Generate and print sample translations."""
    model.eval()
    device = torch.device(config.device)
    
    batch = next(iter(dataloader))
    src = batch['src'][:n_samples].to(device)
    src_texts = batch['src_text'][:n_samples]
    tgt_texts = batch['tgt_text'][:n_samples]
    
    log_info("\n" + "=" * 80)
    log_info("Sample Translations:")
    log_info("=" * 80)
    
    for i in range(min(n_samples, src.size(0))):
        generated = model.generate(
            src[i:i+1], 
            max_len=config.max_len, 
            bos_id=tokenizer.bos_id, 
            eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id
        )
        
        pred_ids = generated[0].tolist()
        pred_text = tokenizer.decode(pred_ids)
        
        log_info(f"\n[Example {i+1}]")
        log_info(f"  Input (DE):      {src_texts[i]}")
        log_info(f"  Ground Truth:    {tgt_texts[i]}")
        log_info(f"  Prediction:      {pred_text}")
        
        if config.debug:
            log_info(f"  Pred IDs:        {pred_ids[:20]}...")
    
    log_info("=" * 80 + "\n")


def main():
    # Load config
    config = get_config()
    
    # Setup logging to file
    log_file = setup_logging(config.log_dir)
    log_info(f"Logging to: {log_file}")
    
    config.print_config()
    
    # Device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    log_info(f"Using device: {device}")
    
    # WandB
    wandb_run = None
    if config.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.__dict__
            )
            log_info(f"WandB initialized: {wandb.run.url}")
        except Exception as e:
            log_info(f"Failed to initialize WandB: {e}")
    
    # Tokenizer
    log_info("\n" + "=" * 60)
    log_info("Loading/Training Tokenizer...")
    log_info("=" * 60)
    
    tokenizer_path = Path(config.tokenizer_path)
    tokenizer_file = tokenizer_path / "tokenizer.json"
    
    if tokenizer_file.exists():
        tokenizer = BPETokenizer.load(str(tokenizer_path))
    else:
        log_info("Training new tokenizer...")
        train_data = load_wmt17("train", max_samples=100000, cache_dir=config.cache_dir)
        
        texts = []
        for item in train_data:
            if 'translation' in item:
                texts.append(item['translation']['de'])
                texts.append(item['translation']['en'])
        
        tokenizer = BPETokenizer.train(texts, vocab_size=config.vocab_size, save_path=str(tokenizer_path))
    
    # Verify tokenizer IDs match config
    log_info(f"\nTokenizer verification:")
    log_info(f"  tokenizer.pad_id = {tokenizer.pad_id} (config: {config.pad_id})")
    log_info(f"  tokenizer.bos_id = {tokenizer.bos_id} (config: {config.bos_id})")
    log_info(f"  tokenizer.eos_id = {tokenizer.eos_id} (config: {config.eos_id})")
    log_info(f"  tokenizer.unk_id = {tokenizer.unk_id} (config: {config.unk_id})")
    
    assert tokenizer.pad_id == config.pad_id, "PAD ID mismatch!"
    assert tokenizer.bos_id == config.bos_id, "BOS ID mismatch!"
    assert tokenizer.eos_id == config.eos_id, "EOS ID mismatch!"
    log_info("  ✓ All token IDs match config")
    
    vocab_size = len(tokenizer)
    log_info(f"  Vocabulary size: {vocab_size}")
    
    # Data
    log_info("\n" + "=" * 60)
    log_info("Loading Data...")
    log_info("=" * 60)
    
    train_loader, val_loader = create_dataloaders(
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_len=config.max_len,
        train_samples=config.train_samples,
        val_samples=config.val_samples,
        cache_dir=config.cache_dir
    )
    
    log_info(f"Train batches: {len(train_loader)}")
    log_info(f"Val batches: {len(val_loader)}")
    
    # Debug first batch
    if config.debug:
        first_batch = next(iter(train_loader))
        debug_batch(first_batch, tokenizer)
    
    # Model
    log_info("\n" + "=" * 60)
    log_info("Creating Model...")
    log_info("=" * 60)
    
    model = Transformer(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        n_layers=config.n_layers,
        dropout=config.dropout,
        max_len=config.max_len,
        pad_idx=config.pad_id,  # Use correct pad_id from config!
        use_rope=config.use_rope
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    log_info(f"Model parameters: {num_params:,}")
    log_info(f"pad_idx in model: {model.pad_idx}")
    
    # Optimizer (Adam or AdamW depending on config)
    optim_cls = AdamW if config.use_adamw else Adam
    optim_kwargs = dict(
        lr=0.0,  # Start with 0 to avoid huge update on first step before scheduler kicks in
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )
    if config.use_adamw:
        optim_kwargs["weight_decay"] = config.adamw_weight_decay
    optimizer = optim_cls(model.parameters(), **optim_kwargs)
    
    # Scheduler
    scheduler = TransformerLRScheduler(
        optimizer=optimizer,
        d_model=config.d_model,
        warmup_steps=config.warmup_steps
    )
    
    
    criterion = nn.CrossEntropyLoss(
        ignore_index=-100,
        label_smoothing=config.label_smoothing
    )
    
    # Mixed precision - BFloat16 does NOT need GradScaler (it has same dynamic range as FP32)
    # GradScaler is only for FP16 which has limited dynamic range
    use_amp = config.use_amp and device.type == 'cuda'
    if use_amp:
        log_info("Mixed precision training enabled (BFloat16 - no scaler needed)")
    
    # Training
    log_info("\n" + "=" * 60)
    log_info("Starting Training...")
    log_info("=" * 60)
    
    # Run-specific checkpoint subfolder (datum_zeit)
    run_checkpoint_dir = Path(config.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"Checkpoints will be saved to: {run_checkpoint_dir}")
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(1, config.epochs + 1):
        log_info(f"\nEpoch {epoch}/{config.epochs}")
        
        train_loss, global_step = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            config=config,
            epoch=epoch,
            global_step=global_step,
            use_amp=use_amp,
            wandb_run=wandb_run,
            val_loader=val_loader,
            tokenizer=tokenizer,
        )
        
        # Validation
        val_loss, bleu_score = validate(
            model, val_loader, criterion, tokenizer, device, config,
            max_batches=config.val_samples // config.batch_size + 1
        )
        
        log_info(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, BLEU: {bleu_score:.2f}")
        
        # Samples
        generate_samples(model, val_loader, tokenizer, config, n_samples=3)
        
        if wandb_run is not None:
            import wandb
            wandb.log({
                'epoch': epoch,
                'epoch/train_loss': train_loss,
                'epoch/val_loss': val_loss,
                'epoch/bleu': bleu_score
            }, step=global_step)
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config.__dict__
            }, run_checkpoint_dir / "best_model.pt")
            log_info(f"Saved best model to {run_checkpoint_dir / 'best_model.pt'} (val_loss: {val_loss:.4f})")
    
    log_info("\n" + "=" * 60)
    log_info("Training Complete!")
    log_info(f"Best validation loss: {best_val_loss:.4f}")
    log_info("=" * 60)
    
    if wandb_run is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
