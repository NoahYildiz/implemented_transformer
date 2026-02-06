"""
Dataset and DataLoader utilities for translation tasks.
Creates proper masks for Transformer training.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple


class TranslationDataset(Dataset):
    """Dataset for sequence-to-sequence translation."""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        src_lang: str = "de",
        tgt_lang: str = "en",
        max_len: int = 128
    ):
        """
        Args:
            data: List of translation pairs
            tokenizer: BPE tokenizer with pad_id, bos_id, eos_id
            src_lang: Source language key
            tgt_lang: Target language key
            max_len: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len
        self.pad_id = tokenizer.pad_id
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Get source and target text
        if 'translation' in item:
            src_text = item['translation'][self.src_lang]
            tgt_text = item['translation'][self.tgt_lang]
        else:
            src_text = item[self.src_lang]
            tgt_text = item[self.tgt_lang]
        
        # Encode source (NO special tokens - encoder input)
        src_ids = self.tokenizer.encode(src_text, add_special_tokens=False)
        if len(src_ids) > self.max_len:
            src_ids = src_ids[:self.max_len]
        
        # Encode target (WITH BOS and EOS)
        # Result: [BOS, tok1, tok2, ..., tokN, EOS]
        tgt_ids = self.tokenizer.encode(tgt_text, add_special_tokens=True)
        if len(tgt_ids) > self.max_len:
            # Truncate but keep EOS at end
            tgt_ids = tgt_ids[:self.max_len - 1] + [self.tokenizer.eos_id]
        
        # Decoder input:  [BOS, tok1, tok2, ..., tokN]  (remove EOS)
        # Labels:         [tok1, tok2, ..., tokN, EOS]  (remove BOS)
        tgt_input = tgt_ids[:-1]
        tgt_labels = tgt_ids[1:]
        
        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt': torch.tensor(tgt_input, dtype=torch.long),
            'labels': torch.tensor(tgt_labels, dtype=torch.long),
            'src_text': src_text,
            'tgt_text': tgt_text
        }


class Collator:
    """
    Collate function for batching with dynamic padding.
    Creates all necessary masks for the Transformer.
    """
    
    def __init__(self, pad_id: int):
        self.pad_id = pad_id
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Find max lengths
        src_max = max(len(item['src']) for item in batch)
        tgt_max = max(len(item['tgt']) for item in batch)
        
        batch_size = len(batch)
        
        # Initialize padded tensors
        src = torch.full((batch_size, src_max), self.pad_id, dtype=torch.long)
        tgt = torch.full((batch_size, tgt_max), self.pad_id, dtype=torch.long)
        labels = torch.full((batch_size, tgt_max), -100, dtype=torch.long)  # -100 = ignore in CE loss
        
        # Keep text for BLEU calculation
        src_texts = []
        tgt_texts = []
        
        for i, item in enumerate(batch):
            src_len = len(item['src'])
            tgt_len = len(item['tgt'])
            
            src[i, :src_len] = item['src']
            tgt[i, :tgt_len] = item['tgt']
            labels[i, :tgt_len] = item['labels']
            
            src_texts.append(item['src_text'])
            tgt_texts.append(item['tgt_text'])
        
        # Create masks
        # src_mask: (batch, 1, 1, src_len) - 1 for valid tokens, 0 for padding
        # Used in encoder self-attention and decoder cross-attention
        src_mask = (src != self.pad_id).unsqueeze(1).unsqueeze(2)
        
        # tgt_mask: (batch, 1, tgt_len, tgt_len) - combines padding + causal mask
        # Used in decoder self-attention
        tgt_pad_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_len)
        tgt_causal_mask = torch.tril(torch.ones(tgt_max, tgt_max, dtype=torch.bool))  # (tgt_len, tgt_len)
        tgt_causal_mask = tgt_causal_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, tgt_len, tgt_len)
        tgt_mask = tgt_pad_mask & tgt_causal_mask  # (batch, 1, tgt_len, tgt_len)
        
        return {
            'src': src,
            'tgt': tgt,
            'labels': labels,
            'src_mask': src_mask,
            'tgt_mask': tgt_mask,
            'src_text': src_texts,
            'tgt_text': tgt_texts
        }


def load_wmt17(
    split: str = "train",
    src_lang: str = "de",
    tgt_lang: str = "en",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> List[Dict]:
    """
    Load WMT17 German-English dataset.
    
    Args:
        split: 'train', 'validation', or 'test'
        src_lang: Source language
        tgt_lang: Target language
        max_samples: Maximum samples to load
        cache_dir: HuggingFace cache directory
    
    Returns:
        List of translation pairs
    """
    import datasets
    
    # Default cache directory
    if cache_dir is None:
        cache_dir = "/data/cat/ws/tosa098h-transformer-ws/noah/cache_huggingface"
    
    dataset = datasets.load_dataset(
        "wmt17",
        f"{src_lang}-{tgt_lang}",
        split=split,
        cache_dir=cache_dir
    )
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    return list(dataset)


def create_dataloaders(
    tokenizer,
    batch_size: int = 32,
    max_len: int = 128,
    train_samples: Optional[int] = None,
    val_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Returns:
        (train_dataloader, val_dataloader)
    """
    # Load data
    train_data = load_wmt17("train", max_samples=train_samples, cache_dir=cache_dir)
    val_data = load_wmt17("validation", max_samples=val_samples, cache_dir=cache_dir)
    
    # Create datasets
    train_dataset = TranslationDataset(train_data, tokenizer, max_len=max_len)
    val_dataset = TranslationDataset(val_data, tokenizer, max_len=max_len)
    
    # Create collator
    collator = Collator(pad_id=tokenizer.pad_id)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    return train_loader, val_loader


def debug_batch(batch: Dict, tokenizer) -> None:
    """Print debug info for a batch."""
    print("\n" + "=" * 60)
    print("BATCH DEBUG INFO")
    print("=" * 60)
    
    print(f"\nShapes:")
    print(f"  src:      {batch['src'].shape}")
    print(f"  tgt:      {batch['tgt'].shape}")
    print(f"  labels:   {batch['labels'].shape}")
    print(f"  src_mask: {batch['src_mask'].shape}")
    print(f"  tgt_mask: {batch['tgt_mask'].shape}")
    
    print(f"\nFirst example:")
    src_ids = batch['src'][0].tolist()
    tgt_ids = batch['tgt'][0].tolist()
    labels = batch['labels'][0].tolist()
    
    print(f"  src_ids:  {src_ids}")
    print(f"  tgt_ids:  {tgt_ids}")
    print(f"  labels:   {labels}")
    
    print(f"\nTokenizer IDs:")
    print(f"  pad_id: {tokenizer.pad_id}")
    print(f"  bos_id: {tokenizer.bos_id}")
    print(f"  eos_id: {tokenizer.eos_id}")
    
    print(f"\nDecoded:")
    print(f"  src: {tokenizer.decode(src_ids)}")
    print(f"  tgt: {tokenizer.decode(tgt_ids)}")
    
    # Check masks
    print(f"\nMask check (first example):")
    src_mask_1d = batch['src_mask'][0, 0, 0, :].tolist()
    valid_src = sum(src_mask_1d)
    print(f"  src_mask valid tokens: {valid_src}/{len(src_mask_1d)}")
    
    tgt_mask_2d = batch['tgt_mask'][0, 0, :, :]
    print(f"  tgt_mask is lower triangular: {torch.allclose(tgt_mask_2d.float(), torch.tril(tgt_mask_2d.float()))}")
    
    # Check labels
    valid_labels = (batch['labels'][0] != -100).sum().item()
    print(f"  valid labels (not -100): {valid_labels}/{len(labels)}")
    
    print("=" * 60 + "\n")
