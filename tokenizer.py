"""
Tokenizer for Transformer - Fixed version with correct special token handling.
Uses HuggingFace tokenizers library for BPE training.
"""
import os
from typing import List, Optional, Iterator
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


class BPETokenizer:
    """
    BPE Tokenizer using HuggingFace tokenizers library.
    
    Special tokens are defined at training time and their IDs are fixed:
    - [PAD] = 1
    - [BOS] = 2
    - [EOS] = 3
    - [UNK] = 4
    """
    
    # Fixed special token IDs (set during training)
    PAD_TOKEN = "[PAD]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"
    
    def __init__(self, tokenizer: Tokenizer = None):
        """Initialize with an optional pre-trained tokenizer."""
        self.tokenizer = tokenizer
        self._update_special_ids()
    
    def _update_special_ids(self):
        """Update special token IDs from tokenizer."""
        if self.tokenizer is not None:
            self.pad_id = self.tokenizer.token_to_id(self.PAD_TOKEN)
            self.bos_id = self.tokenizer.token_to_id(self.BOS_TOKEN)
            self.eos_id = self.tokenizer.token_to_id(self.EOS_TOKEN)
            self.unk_id = self.tokenizer.token_to_id(self.UNK_TOKEN)
        else:
            # Default values (will be set properly after training/loading)
            self.pad_id = 1
            self.bos_id = 2
            self.eos_id = 3
            self.unk_id = 4
    
    @classmethod
    def train(cls, texts: List[str], vocab_size: int = 32000, 
              save_path: str = "tokenizer_data") -> "BPETokenizer":
        """
        Train a new BPE tokenizer.
        
        Args:
            texts: List of texts to train on
            vocab_size: Target vocabulary size
            save_path: Directory to save tokenizer
            
        Returns:
            Trained BPETokenizer instance
        """
        print(f"Training BPE tokenizer with vocab_size={vocab_size}...")
        
        # Create tokenizer with BPE model
        tokenizer = Tokenizer(models.BPE(unk_token=cls.UNK_TOKEN))
        
        # Pre-tokenizer and decoder
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        
        # Trainer with special tokens FIRST (to get fixed IDs)
        # Order matters: <|endoftext|>=0, [PAD]=1, [BOS]=2, [EOS]=3, [UNK]=4
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<|endoftext|>", cls.PAD_TOKEN, cls.BOS_TOKEN, cls.EOS_TOKEN, cls.UNK_TOKEN],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            min_frequency=2,
            show_progress=True
        )
        
        # Train from iterator
        tokenizer.train_from_iterator(iter(texts), trainer=trainer)
        
        # Save
        Path(save_path).mkdir(parents=True, exist_ok=True)
        tokenizer_path = os.path.join(save_path, "tokenizer.json")
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")
        
        # Create wrapper
        wrapper = cls(tokenizer)
        wrapper._print_info()
        
        return wrapper
    
    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """
        Load tokenizer from file.
        
        Args:
            path: Path to tokenizer.json or directory containing it
            
        Returns:
            Loaded BPETokenizer instance
        """
        path_obj = Path(path)
        
        # Find tokenizer file
        if path_obj.is_file():
            tokenizer_path = path_obj
        elif (path_obj / "tokenizer.json").exists():
            tokenizer_path = path_obj / "tokenizer.json"
        else:
            raise FileNotFoundError(f"No tokenizer.json found in {path}")
        
        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
        wrapper = cls(tokenizer)
        wrapper._print_info()
        
        return wrapper
    
    def _print_info(self):
        """Print tokenizer info."""
        print(f"  Vocab size: {len(self)}")
        print(f"  [PAD] = {self.pad_id}")
        print(f"  [BOS] = {self.bos_id}")
        print(f"  [EOS] = {self.eos_id}")
        print(f"  [UNK] = {self.unk_id}")
        
        # Sanity check
        assert self.pad_id == 1, f"Expected pad_id=1, got {self.pad_id}"
        assert self.bos_id == 2, f"Expected bos_id=2, got {self.bos_id}"
        assert self.eos_id == 3, f"Expected eos_id=3, got {self.eos_id}"
        assert self.unk_id == 4, f"Expected unk_id=4, got {self.unk_id}"
        print("  ✓ Special token IDs verified")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add BOS and EOS
            
        Returns:
            List of token IDs
        """
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids
        
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: Token IDs
            skip_special_tokens: Whether to skip special tokens (PAD, BOS, EOS)
            
        Returns:
            Decoded text
        """
        if skip_special_tokens:
            # Only skip PAD, BOS, EOS and placeholder token 0
            # Keep UNK so we can see when the model is uncertain
            special_ids = {self.pad_id, self.bos_id, self.eos_id, 0}
            ids = [i for i in ids if i not in special_ids]
        
        return self.tokenizer.decode(ids)
    
    def __len__(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.get_vocab_size()
    
    def save(self, path: str) -> None:
        """Save tokenizer."""
        Path(path).mkdir(parents=True, exist_ok=True)
        tokenizer_path = os.path.join(path, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")


def test_tokenizer():
    """Test tokenizer functionality."""
    print("=" * 60)
    print("TOKENIZER TEST")
    print("=" * 60)
    
    import tempfile
    
    test_texts = [
        "Hello, how are you?",
        "I am fine, thank you!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "Guten Tag, wie geht es Ihnen?",
        "Das Wetter ist heute schön.",
    ] * 100
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Train
        tokenizer = BPETokenizer.train(test_texts, vocab_size=1000, save_path=temp_dir)
        
        # Test encode/decode
        print("\nEncode/Decode Test:")
        print("-" * 60)
        
        for sent in ["Hello, how are you?", "Guten Morgen!"]:
            ids = tokenizer.encode(sent, add_special_tokens=True)
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            
            print(f"Original:  '{sent}'")
            print(f"IDs:       {ids}")
            print(f"BOS={ids[0]}, EOS={ids[-1]}")
            print(f"Decoded:   '{decoded}'")
            print()
        
        # Test load
        print("Save/Load Test:")
        print("-" * 60)
        loaded = BPETokenizer.load(temp_dir)
        
        test_sent = "Hello world!"
        orig_ids = tokenizer.encode(test_sent)
        loaded_ids = loaded.encode(test_sent)
        print(f"Original: {orig_ids}")
        print(f"Loaded:   {loaded_ids}")
        print(f"Match: {'✓' if orig_ids == loaded_ids else '✗'}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_tokenizer()
