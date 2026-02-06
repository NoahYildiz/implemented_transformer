"""
Transformer Model - Exact implementation following "Attention is All You Need".

Architecture:
- d_model = 512
- n_heads = 8 (d_k = d_v = 64)
- d_ff = 2048
- N = 6 (encoder and decoder layers)
- Dropout = 0.1
- Sinusoidal positional encoding
- Label smoothing = 0.1
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE)."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # RoPE is applied to pairs of coordinates, so we need d_model/2 frequencies
        # If head dimension is d_k, we usually apply it to d_k
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        # Concatenate sin/cos arguments
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('emb', emb)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return self.emb[:seq_len, :].to(device)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: (batch, n_heads, seq_len, d_k)
        pos_emb: (seq_len, d_k)
    """
    # Reshape pos_emb for broadcasting: (1, 1, seq_len, d_k)
    # Note: We assume pos_emb has shape (seq_len, d_k)
    pos_emb = pos_emb.unsqueeze(0).unsqueeze(0)
    return (x * pos_emb.cos()) + (rotate_half(x) * pos_emb.sin())


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, rope_module: Optional[RotaryEmbedding] = None):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.rope_module = rope_module
        
        # Linear projections without bias (as in original paper)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: (batch, query_len, d_model)
            key: (batch, key_len, d_model)
            value: (batch, key_len, d_model)
            mask: (batch, 1, 1, key_len) or (batch, 1, query_len, key_len)
        Returns:
            (batch, query_len, d_model)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape to (batch, n_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply RoPE if enabled (usually only for self-attention)
        if self.rope_module is not None:
            # Get positions based on sequence length
            seq_len_q = Q.size(2)
            seq_len_k = K.size(2)
            
            # Fetch embeddings
            pos_emb_q = self.rope_module(seq_len_q, Q.device)
            pos_emb_k = self.rope_module(seq_len_k, K.device)
            
            # Apply rotation
            Q = apply_rotary_pos_emb(Q, pos_emb_q)
            K = apply_rotary_pos_emb(K, pos_emb_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(context)


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single Transformer Encoder Layer with Pre-LN (more stable training)."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, rope_module: Optional[RotaryEmbedding] = None):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, rope_module=rope_module)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model) # Pre-LN
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN: Normalize BEFORE attention/ffn
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out = self.self_attn(normed, normed, normed, src_mask)
        x = x + self.dropout1(attn_out)
        
        # FFN with residual
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout2(ffn_out)
        
        return x


class DecoderLayer(nn.Module):
    """Single Transformer Decoder Layer with Pre-LN (more stable training)."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, rope_module: Optional[RotaryEmbedding] = None):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, rope_module=rope_module)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, rope_module=None) # No RoPE for cross-attn
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model) # Pre-LN
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN: Normalize BEFORE attention/ffn
        # Masked self-attention
        normed = self.norm1(x)
        attn_out = self.self_attn(normed, normed, normed, tgt_mask)
        x = x + self.dropout1(attn_out)
        
        # Cross-attention
        normed = self.norm2(x)
        attn_out = self.cross_attn(normed, encoder_out, encoder_out, src_mask)
        x = x + self.dropout2(attn_out)
        
        # FFN
        normed = self.norm3(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout3(ffn_out)
        
        return x


class Transformer(nn.Module):
    """
    Transformer model for sequence-to-sequence tasks.
    
    Default hyperparameters from "Attention is All You Need":
    - d_model = 512
    - n_heads = 8
    - d_ff = 2048
    - n_layers = 6
    - dropout = 0.1
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        n_layers: int = 6,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_idx: int = 0,
        share_embeddings: bool = True,
        use_rope: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.use_rope = use_rope
        
        # Embeddings
        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        if share_embeddings:
            self.tgt_embed = self.src_embed
        else:
            self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # Positional encoding
        if not use_rope:
            self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        else:
            # RoPE applies to the head dimension (d_k), NOT d_model
            d_k = d_model // n_heads
            self.rope_module = RotaryEmbedding(d_k, max_len)
            self.dropout_emb = nn.Dropout(dropout)
        
        rope_to_pass = self.rope_module if use_rope else None

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, rope_module=rope_to_pass) for _ in range(n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, rope_module=rope_to_pass) for _ in range(n_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)
        
        # Output projection (tied with embedding weights)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.src_embed.weight
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters properly.
        
        - Embeddings: Normal(0, 0.02) - standard for transformers
        - Linear layers: Xavier uniform
        - LayerNorm: default (weight=1, bias=0)
        """
        for name, p in self.named_parameters():
            if 'embed' in name:
                # Embeddings get normal distribution
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif p.dim() > 1:
                # Linear layers get Xavier
                nn.init.xavier_uniform_(p)
    
    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create source padding mask.
        
        Args:
            src: (batch, src_len)
        Returns:
            (batch, 1, 1, src_len) - 1 for valid, 0 for padding
        """
        mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
    
    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Create target mask combining padding mask and causal mask.
        
        Args:
            tgt: (batch, tgt_len)
        Returns:
            (batch, 1, tgt_len, tgt_len)
        """
        batch_size, tgt_len = tgt.size()
        
        # Padding mask: (batch, 1, 1, tgt_len)
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Causal mask: (1, 1, tgt_len, tgt_len)
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        
        # Combine: (batch, 1, tgt_len, tgt_len)
        mask = pad_mask & causal_mask
        return mask
    
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """Encode source sequence."""
        # Embed and add positional encoding
        x = self.src_embed(src) * math.sqrt(self.d_model)
        
        if not self.use_rope:
             x = self.pos_encoding(x)
        else:
             x = self.dropout_emb(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return self.encoder_norm(x)
    
    def decode(self, tgt: torch.Tensor, encoder_out: torch.Tensor,
               src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """Decode target sequence."""
        # Embed and add positional encoding
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)

        if not self.use_rope:
             x = self.pos_encoding(x)
        else:
             x = self.dropout_emb(x)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)
        
        return self.decoder_norm(x)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            src: Source token IDs (batch, src_len)
            tgt: Target token IDs for decoder input (batch, tgt_len)
            src_mask: Optional source mask (batch, 1, 1, src_len)
            tgt_mask: Optional target mask (batch, 1, tgt_len, tgt_len)
        
        Returns:
            Logits (batch, tgt_len, vocab_size)
        """
        # Create masks if not provided
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)
        
        # Encode
        encoder_out = self.encode(src, src_mask)
        
        # Decode
        decoder_out = self.decode(tgt, encoder_out, src_mask, tgt_mask)
        
        # Project to vocabulary
        logits = self.output_proj(decoder_out)
        
        return logits
    
    @torch.no_grad()
    def generate(self, src: torch.Tensor, max_len: int, bos_id: int, eos_id: int,
                 pad_id: int = None) -> torch.Tensor:
        """
        Greedy decoding for inference.
        
        Args:
            src: Source token IDs (batch, src_len)
            max_len: Maximum generation length
            bos_id: BOS token ID
            eos_id: EOS token ID
            pad_id: PAD token ID (defaults to self.pad_idx)
        
        Returns:
            Generated token IDs (batch, gen_len)
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        if pad_id is None:
            pad_id = self.pad_idx
        
        # Encode source once
        src_mask = self.make_src_mask(src)
        encoder_out = self.encode(src, src_mask)
        
        # Start with BOS token
        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        
        # Track which sequences are finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_len - 1):
            # Create target mask
            tgt_mask = self.make_tgt_mask(generated)
            
            # Decode
            decoder_out = self.decode(generated, encoder_out, src_mask, tgt_mask)
            
            # Get logits for last position only
            logits = self.output_proj(decoder_out[:, -1])  # (batch, vocab_size)
            
            # Greedy: select highest probability token
            next_token = logits.argmax(dim=-1)  # (batch,)
            
            # Replace with PAD for already finished sequences
            next_token = next_token.masked_fill(finished, pad_id)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            
            # Update finished status
            finished = finished | (next_token == eos_id)
            
            # Stop if all sequences are finished
            if finished.all():
                break
        
        return generated
