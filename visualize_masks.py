import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

def print_matrix(matrix, row_tokens, col_tokens, name):
    """Hübsche Ausgabe einer Attention Maske."""
    # Konvertiere bool/int maske zu Symbolen
    # 1/True = Sichtbar (.), 0/False = Maskiert (#)
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()
    
    print(f"\n=== {name} ===")
    print("Legende: '.' = Sichtbar (Attention möglich), '#' = Blockiert (Maskiert)")
    
    # Header formatieren
    print("      " + " ".join([f"{t:>3}" for t in col_tokens]))
    
    for i, row_tok in enumerate(row_tokens):
        row_str = f"{row_tok:>3} |"
        for j, col_tok in enumerate(col_tokens):
            if i < matrix.shape[0] and j < matrix.shape[1]:
                val = matrix[i, j]
                symbol = "." if val > 0 else "#"
                row_str += f" {symbol:>3}"
        print(row_str)

def visualize_masks():
    print("Generiere Dummy Batch für Masken-Check...\n")
    
    # Beispiel Daten
    # Source: "Hallo Welt" (kurz) -> Padding nötig
    # Target: "[BOS] Hello World [EOS]"
    
    max_len = 6
    pad_id = 1
    
    # Dummy Token IDs
    # src: [10, 11, PAD, PAD] (Länge 4, Pad ab index 2)
    src = torch.tensor([[10, 11, pad_id, pad_id]]) 
    src_tokens = ["Hal", "lo", "PAD", "PAD"]
    
    # tgt: [BOS, 20, 21, EOS, PAD] (Länge 5)
    tgt = torch.tensor([[2, 20, 21, 3, pad_id]])
    tgt_tokens = ["BOS", "Hel", "lo", "EOS", "PAD"]
    
    # --- 1. Source Mask ---
    # Shape: (batch, 1, 1, src_len)
    # Erlaubt Attention auf alles was NICHT Pad ist
    src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2)
    
    # Wir nehmen die letzte Dimension für die Visualisierung (Key-Dimension)
    # Da es für alle Queries gleich ist, zeigen wir 1D Zeile
    print(f"Source Mask Shape: {src_mask.shape}")
    print(f"Maskiert Padding Tokens im Encoder.")
    mask_view = src_mask[0, 0, 0].int().numpy()
    
    print("      " + " ".join([f"{t:>3}" for t in src_tokens]))
    print("Mask: " + " ".join([f"{'.':>3}" if x else f"{'#':>3}" for x in mask_view]))
    print("(Padding Tokens 'PAD' werden blockiert)")

    
    # --- 2. Target Mask (Der spannende Teil) ---
    tgt_len = tgt.size(1)
    
    # Padding Mask (Wie bei Source)
    tgt_pad_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2) # (1, 1, 1, 5)
    
    # Causal Mask (Dreieck)
    tgt_causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool)) # (5, 5)
    tgt_causal_mask = tgt_causal_mask.unsqueeze(0).unsqueeze(1) # (1, 1, 5, 5)
    
    # Kombiniert
    tgt_mask = tgt_pad_mask & tgt_causal_mask
    
    print(f"\n{'='*40}")
    print(f"Target Mask Shape: {tgt_mask.shape}")
    print("Kombiniert Causal Mask (Kein Blick in Zukunft) UND Padding Mask.")
    
    # Visualisiere die 5x5 Matrix
    matrix = tgt_mask[0, 0].int()
    print_matrix(matrix, tgt_tokens, tgt_tokens, "Target Self-Attention Mask")
    
    print("\nErklärung:")
    print("1. Zeile (BOS): Darf nur sich selbst sehen (.)")
    print("2. Zeile (Hel): Darf BOS und sich selbst sehen (. .)")
    print("4. Zeile (EOS): Sieht den ganzen Satz (. . . .)")
    print("5. Zeile (PAD): Ist Padding, darf nix sehen (auch wenn Causal Mask erlaubt)")
    print("Spalte 5 (PAD): Niemand darf PAD anschauen (# in letzter Spalte)")

if __name__ == "__main__":
    visualize_masks()
