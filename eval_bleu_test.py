"""
Evaluate best checkpoint(s) on the full WMT17 test set and report BLEU.

Supports comparing:
  - Model with RoPE (newest): e.g. best_model.pt
  - Model without RoPE (previous): e.g. best_model_no_rope.pt


/data/cat/ws/tosa098h-transformer-ws/noah/checkpoints/adamw_rope

Usage:
  # Evaluate both if you have two checkpoints (RoPE + no RoPE):
  python eval_bleu_test.py --checkpoint_rope /data/cat/ws/tosa098h-transformer-ws/noah/checkpoints/adamw_rope/best_model.pt \\
                           --checkpoint_no_rope /data/cat/ws/tosa098h-transformer-ws/noah/checkpoints/adamw_no_rope/best_model.pt

  # Evaluate only the current best (e.g. with RoPE):
  python eval_bleu_test.py --checkpoint_rope /path/to/checkpoints/best_model.pt

  # Evaluate only the no-RoPE checkpoint:
  python eval_bleu_test.py --checkpoint_no_rope /data/cat/ws/tosa098h-transformer-ws/noah/checkpoints/no_rope/best_model.pt
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from dataclasses import fields

from config import Config
from model import Transformer
from tokenizer import BPETokenizer
from dataset import load_wmt17, TranslationDataset, Collator
from evaluate import compute_bleu


def parse_args():
    p = argparse.ArgumentParser(description="BLEU on full WMT17 test set for best checkpoints.")
    p.add_argument(
        "--checkpoint_rope",
        type=str,
        default=None,
        help="Path to best checkpoint WITH RoPE (newest model).",
    )
    p.add_argument(
        "--checkpoint_no_rope",
        type=str,
        default=None,
        help="Path to best checkpoint WITHOUT RoPE (previous model).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/data/cat/ws/tosa098h-transformer-ws/noah/checkpoints",
        help="Default directory for checkpoints if only relative names given.",
    )
    p.add_argument(
        "--tokenizer_path",
        type=str,
        default="./tokenizer_data",
        help="Path to tokenizer directory.",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default="/data/cat/ws/tosa098h-transformer-ws/noah/cache_huggingface",
        help="HuggingFace cache for WMT17.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu).",
    )
    p.add_argument(
        "--num_examples",
        type=int,
        default=30,
        help="Number of qualitative translation examples to print per model.",
    )
    return p.parse_args()


def config_from_checkpoint(ckpt: dict) -> Config:
    """Build Config from saved checkpoint dict (only known fields)."""
    config_dict = ckpt["config"]
    valid = {f.name for f in fields(Config)}
    filtered = {k: v for k, v in config_dict.items() if k in valid}
    return Config(**filtered)


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> tuple:
    """
    Load model and config from a checkpoint file.
    Returns (model, config).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = config_from_checkpoint(ckpt)

    model = Transformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        n_layers=config.n_layers,
        dropout=config.dropout,
        max_len=config.max_len,
        pad_idx=config.pad_id,
        use_rope=config.use_rope,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, config


@torch.no_grad()
def run_bleu_on_test_set(model, test_loader, tokenizer, config, device, num_examples: int = 0):
    """
    Run model on full test set, collect hypotheses and references.
    Returns (bleu_stats, examples) where examples is a list of (src, ref, hyp) for the first num_examples.
    """
    hypotheses = []
    references = []
    examples = []  # (src_text, ref_text, hyp_text)
    for batch in test_loader:
        src = batch["src"].to(device)
        src_texts = batch["src_text"]
        tgt_texts = batch["tgt_text"]
        batch_size = src.size(0)
        for i in range(batch_size):
            generated = model.generate(
                src[i : i + 1],
                max_len=config.max_len,
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                pad_id=tokenizer.pad_id,
            )
            pred_text = tokenizer.decode(generated[0].tolist())
            hypotheses.append(pred_text)
            references.append(tgt_texts[i])
            if num_examples > 0 and len(examples) < num_examples:
                examples.append((src_texts[i], tgt_texts[i], pred_text))
    return compute_bleu(references, hypotheses), examples


def print_qualitative_examples(label: str, examples: list, num: int = 30):
    """Print qualitative translation examples (source, reference, hypothesis)."""
    n = min(num, len(examples))
    if n == 0:
        return
    print(f"\n--- Qualitative Beispiele ({label}, {n} Stück) ---")
    for i, (src, ref, hyp) in enumerate(examples[:n], 1):
        print(f"\n  [{i}] DE:  {src}")
        print(f"      EN (Ref): {ref}")
        print(f"      EN (Hyp): {hyp}")
    print()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Resolve checkpoint paths (optional default: output_dir/best_model.pt and best_model_no_rope.pt)
    output_dir = Path(args.output_dir)
    checkpoint_rope = args.checkpoint_rope
    checkpoint_no_rope = args.checkpoint_no_rope
    if checkpoint_rope is not None and not Path(checkpoint_rope).is_absolute():
        checkpoint_rope = str(output_dir / checkpoint_rope)
    if checkpoint_no_rope is not None and not Path(checkpoint_no_rope).is_absolute():
        checkpoint_no_rope = str(output_dir / checkpoint_no_rope)

    if not checkpoint_rope and not checkpoint_no_rope:
        # Default: try both standard names
        checkpoint_rope = str(output_dir / "best_model.pt")
        checkpoint_no_rope = str(output_dir / "best_model_no_rope.pt")
        if not Path(checkpoint_rope).exists():
            checkpoint_rope = None
        if not Path(checkpoint_no_rope).exists():
            checkpoint_no_rope = None
        if not checkpoint_rope and not checkpoint_no_rope:
            raise FileNotFoundError(
                "No checkpoints given and none found at "
                f"{output_dir}/best_model.pt or {output_dir}/best_model_no_rope.pt"
            )

    # Tokenizer
    tokenizer = BPETokenizer.load(args.tokenizer_path)

    # Test set
    print("Loading WMT17 test set...")
    test_data = load_wmt17("test", cache_dir=args.cache_dir)
    test_dataset = TranslationDataset(
        test_data,
        tokenizer,
        src_lang="de",
        tgt_lang="en",
        max_len=192,
    )
    collator = Collator(pad_id=tokenizer.pad_id)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    print(f"  Test samples: {len(test_dataset)}")

    results = []
    num_examples = args.num_examples

    def eval_one(label: str, path: str):
        if not path or not Path(path).exists():
            print(f"  [{label}] Skip (file not found): {path}")
            return
        print(f"\n[{label}] Loading {path} ...")
        model, config = load_model_from_checkpoint(path, device)
        print(f"  use_rope={config.use_rope}, use_adamw={config.use_adamw}, max_len={config.max_len}")
        print(f"  Running inference on full test set ...")
        stats, examples = run_bleu_on_test_set(
            model, test_loader, tokenizer, config, device, num_examples=num_examples
        )
        results.append((label, stats))
        print(f"  BLEU: {stats['bleu']:.2f}  (BP: {stats['brevity_penalty']:.4f})")
        print_qualitative_examples(label, examples, num=num_examples)

    if checkpoint_rope:
        eval_one("RoPE (newest)", checkpoint_rope)
    if checkpoint_no_rope:
        eval_one("no RoPE (previous)", checkpoint_no_rope)

    # Summary
    print("\n" + "=" * 60)
    print("BLEU on full WMT17 test set")
    print("=" * 60)
    for label, stats in results:
        print(f"  {label}:  BLEU = {stats['bleu']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
