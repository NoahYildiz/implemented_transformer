# Implementing Transformer

Einfache, saubere Transformer-Implementierung exakt nach **"Attention is All You Need"** (Vaswani et al., 2017).

## Architektur

Standardwerte entsprechen dem Originalpaper:

| Parameter | Wert |
|-----------|------|
| d_model | 512 |
| n_heads | 8 |
| d_ff | 2048 |
| n_layers | 6 (Encoder & Decoder) |
| dropout | 0.1 |
| Label Smoothing | 0.1 |


## Dateien

```
transformer2/
├── model.py       # Transformer-Architektur
├── tokenizer.py   # BPE Tokenizer
├── dataset.py     # Dataset und DataLoader
├── train.py       # Training-Script
└── README.md
```

## Training starten

```bash
# Einfaches Training
python train.py --epochs 20 --batch_size 32

# Mit WandB
python train.py --use_wandb --wandb_project my_project

# Kleinerer Test
python train.py --train_samples 10000 --epochs 5

# Vollständiges Training (wie im Paper)
python train.py \
    --d_model 512 \
    --n_heads 8 \
    --d_ff 2048 \
    --n_layers 6 \
    --batch_size 64 \
    --epochs 30 \
    --warmup_steps 4000 \
    --use_wandb
```

## Tokenizer

Der Tokenizer wird automatisch trainiert beim ersten Lauf und in `tokenizer.json` gespeichert. Beim nächsten Lauf wird er geladen.

```python
from tokenizer import BPETokenizer

# Manuelles Training
tokenizer = BPETokenizer(vocab_size=32000)
tokenizer.fit(texts)
tokenizer.save("tokenizer.json")

# Laden
tokenizer = BPETokenizer.load("tokenizer.json")

# Oder: automatisch laden oder trainieren
tokenizer = BPETokenizer.load_or_train("tokenizer.json", texts, vocab_size=32000)
```

## WandB Metriken

- `train/loss` - Training Loss pro 100 Steps
- `train/lr` - Learning Rate
- `val/loss` - Validation Loss alle 5000 Steps
- `epoch/train_loss` - Durchschnittlicher Training Loss pro Epoche
- `epoch/val_loss` - Validation Loss am Epochenende


## Kurze Info: Dieses Repo wurde im Laufe des Implementing Transformers Modul an der HHU entwickelt.
## Ein vollständiger Trainingsrun mit der base config auf einer H100er hat ca. 5 Stunden gedauert.





