"""
Evaluation utilities: BLEU score calculation using sacrebleu.
"""
import torch
from typing import List, Dict, Tuple, Optional
import sacrebleu


def compute_bleu(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Compute BLEU score using sacrebleu (corpus-level).
    
    Args:
        references: List of reference translations
        hypotheses: List of generated translations
    
    Returns:
        Dictionary with BLEU scores
    """
    # sacrebleu expects references as list of lists (for multiple references per hypothesis)
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    
    return {
        'bleu': bleu.score,
        'bleu-1': bleu.precisions[0] if len(bleu.precisions) > 0 else 0,
        'bleu-2': bleu.precisions[1] if len(bleu.precisions) > 1 else 0,
        'bleu-3': bleu.precisions[2] if len(bleu.precisions) > 2 else 0,
        'bleu-4': bleu.precisions[3] if len(bleu.precisions) > 3 else 0,
        'brevity_penalty': bleu.bp
    }


@torch.no_grad()
def validate(model, dataloader, criterion, tokenizer, device, config, max_batches: Optional[int] = None) -> Tuple[float, float]:
    """
    Calculate validation loss and BLEU score.
    
    Args:
        model: Transformer model
        dataloader: Validation dataloader
        criterion: Loss function
        tokenizer: Tokenizer for decoding
        device: Torch device (str or torch.device)
        config: Configuration object
        max_batches: Limit number of batches for faster validation (optional)
        
    Returns:
        (avg_loss, bleu_score)
    """
    model.eval()
    if isinstance(device, str):
        device = torch.device(device)
        
    total_loss = 0
    total_tokens = 0
    
    hypotheses = []
    references = []
    
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        labels = batch['labels'].to(device)
        src_mask = batch['src_mask'].to(device)
        tgt_mask = batch['tgt_mask'].to(device)
        tgt_texts = batch['tgt_text']
        
        # Loss
        logits = model(src, tgt, src_mask, tgt_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        num_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        
        # Generate for BLEU (limited samples)
        if len(hypotheses) < config.bleu_samples:
            n_to_generate = min(src.size(0), config.bleu_samples - len(hypotheses))
            for i in range(n_to_generate):
                generated = model.generate(
                    src[i:i+1],
                    max_len=config.max_len,
                    bos_id=tokenizer.bos_id,
                    eos_id=tokenizer.eos_id,
                    pad_id=tokenizer.pad_id
                )
                pred_text = tokenizer.decode(generated[0].tolist())
                hypotheses.append(pred_text)
                references.append(tgt_texts[i])
    
    # BLEU score
    if len(hypotheses) > 0:
        bleu_stats = compute_bleu(references, hypotheses)
        bleu_score = bleu_stats['bleu']
    else:
        bleu_score = 0.0
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss, bleu_score
