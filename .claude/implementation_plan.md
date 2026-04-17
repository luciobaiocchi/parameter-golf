# Parameter Golf: Guida Sperimentale XSA + Testing Framework

## Indice

1. [Contesto e Obiettivo](#contesto)
2. [Come Funziona il Pipeline del Record](#pipeline)
3. [XSA: Implementazione](#xsa-implementazione)
4. [Matrice Completa degli Esperimenti](#esperimenti)
5. [Framework di Testing](#framework)
6. [Setup Ambiente](#setup)
7. [Script di Automazione](#automazione)
8. [Budget e Timeline](#budget)
9. [Criteri di Decisione](#decisioni)
10. [Preparazione della PR](#pr)

---

## 1. Contesto e Obiettivo <a name="contesto"></a>

### La Competizione

Parameter Golf richiede un modello linguistico in un artefatto ≤16 MB, trainato in ≤10 min su 8×H100 SXM, valutato in bits-per-byte (BPB) su FineWeb validation. Deadline: 30 aprile 2026.

### Record Attuale

- **BPB**: 1.0810 (3-seed mean, std 0.0002)
- **Stack**: SP8192 + 11L×512d×8H + depth recurrence 3-layer + parallel residuals + QK-Gain 5.25 + MuonEq-R + GPTQ SDClip int6 + Legal TTT
- **Soglia per nuova PR**: ≤ 1.076 BPB (delta ≥ 0.005)

### Il Nostro Angolo

Testare **Exclusive Self Attention (XSA)** in combinazione con il full stack del record.
Nessuno ha ancora testato XSA + depth recurrence + TTT insieme.

---

## 2. Come Funziona il Pipeline del Record <a name="pipeline"></a>

### Training (588s su 8×H100)

```
Step 1-682 (0-15%):    Training bf16, no QAT, no depth recurrence
Step 683 (15%):         QAT si attiva (Late QAT con STE)
Step 1592 (35%):        Depth recurrence si attiva (layer 3-5 looped)
Step 1274-4550 (28-100%): Warmdown lineare del LR a 0
Tutto il training:      EMA decay 0.9965, WD=0.095
```

### Quantizzazione Post-Training

```
1. Prendi il modello EMA (non l'ultimo checkpoint)
2. GPTQ full-Hessian con SDClip:
   - Matrici attn/MLP → int6 (clip = 12.85 × std(row))
   - Embedding → int8 (clip = 20.0 × std(row))
3. Byte-shuffle + Brotli-11 compression
4. Wrap in codice LZMA (~16.6 KB)
```

### Eval (500s su 8×H100)

```
1. Sliding window eval (stride 64, causale)
2. TTT: chunk 32K token, score-first, SGD (lr=0.005, momentum=0.9)
   - Per ogni chunk: score sotto no_grad → train 3 epoche → next chunk
   - Cosine LR decay across chunks
```

### Punti Chiave per XSA

- Il training usa bf16 con QAT che simula int6 — XSA opera in bf16, ma deve rimanere stabile anche quando i pesi vengono quantizzati
- Il modello EMA viene usato per la quantizzazione — XSA deve funzionare bene con pesi "smoothed"
- Il TTT modifica i pesi a eval time — XSA deve essere stabile anche con pesi che cambiano durante l'eval

---

## 3. XSA: Implementazione <a name="xsa-implementazione"></a>

### Il Codice (2 righe)

Nell'attention forward del modello, dopo `scaled_dot_product_attention`:

```python
# === PRIMA (attenzione standard) ===
# Y shape: (B, H, T, d_head) dove d_head = 64
Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

# === DOPO (con XSA) ===
Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

# XSA: proiezione ortogonale — rimuovi componente self-value
Vn = F.normalize(V, dim=-1)                                # (B, H, T, d_head)
Y = Y - (Y * Vn).sum(dim=-1, keepdim=True) * Vn            # (B, H, T, d_head)
```

### Dove Inserirlo in train_gpt.py

Cerca la classe `CausalSelfAttention` o equivalente. Il punto di inserimento è tra
l'output di `scaled_dot_product_attention` e la proiezione output `Wo`.

```python
class CausalSelfAttention(nn.Module):
    def forward(self, x):
        B, T, C = x.size()

        # Proiezioni Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, C // self.n_head).transpose(1, 2)

        # Applica QK-Gain
        q = q * self.qk_gain  # scalare learnable per testa

        # Attenzione standard
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # ============ XSA INSERT HERE ============
        if self.use_xsa:
            vn = F.normalize(v, dim=-1)
            # Se GQA (4 KV-heads, 8 Q-heads), espandi v per matchare
            if self.n_kv_head != self.n_head:
                vn = vn.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
            y = y - (y * vn).sum(dim=-1, keepdim=True) * vn
        # ==========================================

        # Reshape e proiezione output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
```

### Attenzione: GQA (Grouped Query Attention)

Il record usa 8 teste Q e 4 teste KV (GQA). Ogni testa KV serve 2 teste Q.
V ha shape `(B, 4, T, 64)` ma Y ha shape `(B, 8, T, 64)`.
Devi espandere Vn per matchare: `vn = vn.repeat_interleave(2, dim=1)`.

### Configurazione Selettiva per Layer

Per testare XSA solo su certi layer, aggiungi un flag per-layer:

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # ... (esistente)
        self.use_xsa = layer_idx in config.xsa_layers  # set() di indici
```

E nel config:

```python
# Varianti da testare:
xsa_layers = set(range(11))           # XSA-all
xsa_layers = {3, 4, 5}                # XSA solo layer ricorrenti
xsa_layers = set(range(7))            # XSA no-parallel-residuals
xsa_layers = set(range(6, 11))        # XSA solo layer profondi
xsa_layers = set()                    # Baseline (no XSA)
```

---

## 4. Matrice Completa degli Esperimenti <a name="esperimenti"></a>

### Legenda

- **Proxy**: 1500 step, 1×H100, seed 42, ~3 min, ~$0.15
- **Full-1**: 4550 step, 1×H100, seed 42, ~10 min, ~$0.50
- **Full-8**: 4550 step, 8×H100, 3 seed, ~10 min × 3, ~$4

### Blocco 1: Dove Applicare XSA (8 run proxy)

| Run | XSA Config | Ipotesi | Note |
|-----|-----------|---------|------|
| 1 | Tutti i layer (0-10) | Baseline XSA | Come PR #1630 |
| 2 | Solo ricorrenti (3,4,5) | Similarity bias si accumula nei loop | **Alta priorità** |
| 3 | Solo NON ricorrenti (0,1,2,6,7,8,9,10) | Test inverso del run 2 | |
| 4 | Solo pre-parallel (0-6) | Parallel res. complicano XSA | |
| 5 | Solo parallel (7-10) | Test inverso del run 4 | |
| 6 | Solo profondi (6-10) | Paper: bias cresce con depth | **Alta priorità** |
| 7 | Solo superficiali (0-5) | Test inverso del run 6 | |
| 8 | Nessuno (baseline) | Riferimento | **Fare per primo** |

**Decisione**: prendi la config con loss più bassa e il delta più grande vs baseline.

### Blocco 2: XSA + QK-Gain (6 run proxy)

Usa la miglior config XSA dal Blocco 1.

| Run | QK-Gain | Ipotesi |
|-----|---------|---------|
| 9 | 3.0 | XSA riduce necessità di gain alto |
| 10 | 4.0 | |
| 11 | 5.25 | Stesso del record |
| 12 | 6.0 | XSA libera spazio per gain più alto |
| 13 | 7.0 | |
| 14 | 8.0 | Molto aggressivo |

### Blocco 3: XSA + Depth Recurrence (6 run proxy)

| Run | Config | Ipotesi |
|-----|--------|---------|
| 15 | XSA-best + DR 3-layer (3,4,5) @ frac=0.35 | Combo record |
| 16 | XSA-recurrent + DR 3-layer @ frac=0.35 | XSA solo dove serve |
| 17 | XSA-best + DR 4-layer (2,3,4,5) @ frac=0.35 | Più virtual layers |
| 18 | XSA-best + DR 3-layer @ frac=0.25 | Attiva ricorrenza prima |
| 19 | XSA-best + DR 3-layer @ frac=0.30 | |
| 20 | XSA-best + DR 3-layer @ frac=0.45 | Attiva dopo |

### Blocco 4: Hyperparameter Sweep con XSA (12 run proxy)

Usa la miglior combo da Blocco 1-3.

| Run | Parametro | Valore | Record |
|-----|-----------|--------|--------|
| 21 | LeakyReLU slope | 0.3 | 0.5 |
| 22 | LeakyReLU slope | 0.4 | 0.5 |
| 23 | LeakyReLU slope | 0.6 | 0.5 |
| 24 | Partial RoPE | 24/64 | 16/64 |
| 25 | Partial RoPE | 32/64 | 16/64 |
| 26 | EMA decay | 0.995 | 0.9965 |
| 27 | EMA decay | 0.997 | 0.9965 |
| 28 | EMA decay | 0.998 | 0.9965 |
| 29 | Warmdown fraction | 0.65 | 0.72 |
| 30 | Warmdown fraction | 0.80 | 0.72 |
| 31 | QAT timing | 0.10 | 0.15 |
| 32 | QAT timing | 0.20 | 0.15 |

### Blocco 5: TTT Optimization (4 run, full-1)

| Run | TTT Config | Record |
|-----|-----------|--------|
| 33 | lr=0.003, 3 epoche, 32K chunk | lr=0.005 |
| 34 | lr=0.008, 3 epoche, 32K chunk | lr=0.005 |
| 35 | lr=0.005, 4 epoche, 32K chunk | 3 epoche |
| 36 | lr=0.005, 3 epoche, 16K chunk | 32K chunk |

### Blocco 6: Combinazione Finale e Validazione (6 run, full-8)

| Run | Config | GPU |
|-----|--------|-----|
| 37-39 | Miglior combo, seed 42/314/999 | 8×H100 |
| 40-42 | Seconda miglior combo, seed 42/314/999 | 8×H100 |

---

## 5. Framework di Testing <a name="framework"></a>

### Struttura Directory

```
parameter-golf/
├── experiments/
│   ├── configs/           # JSON config per ogni run
│   ├── logs/              # Output del training
│   ├── results/           # CSV aggregato
│   └── analysis/          # Notebook/script di analisi
├── train_gpt.py           # Training script (modificato con XSA)
├── run_experiment.sh      # Wrapper per singolo esperimento
├── run_sweep.py           # Orchestratore sweep
└── analyze_results.py     # Aggregazione e analisi
```

### Config JSON per Esperimento

```json
{
    "run_id": "xsa_b1_r02_recurrent_only",
    "block": 1,
    "description": "XSA solo su layer ricorrenti (3,4,5)",

    "architecture": {
        "n_layers": 11,
        "d_model": 512,
        "n_heads": 8,
        "n_kv_heads": 4,
        "mlp_expansion": 4,
        "activation": "leaky_relu_sq",
        "leaky_slope": 0.5,
        "rope_dims": 16,
        "logit_softcap": 30.0
    },

    "xsa": {
        "enabled": true,
        "layers": [3, 4, 5]
    },

    "depth_recurrence": {
        "enabled": true,
        "layers": [3, 4, 5],
        "activation_frac": 0.35
    },

    "parallel_residuals": {
        "enabled": true,
        "from_layer": 7
    },

    "training": {
        "qk_gain_init": 5.25,
        "weight_decay": 0.095,
        "max_lr": 0.022,
        "ema_decay": 0.9965,
        "warmdown_frac": 0.72,
        "qat_start_frac": 0.15,
        "total_steps": 1500,
        "seed": 42
    },

    "quantization": {
        "method": "gptq_sdclip",
        "matrix_bits": 6,
        "matrix_clip_k": 12.85,
        "embedding_bits": 8,
        "embedding_clip_k": 20.0,
        "compression": "brotli-11"
    },

    "ttt": {
        "enabled": false,
        "lr": 0.005,
        "momentum": 0.9,
        "epochs": 3,
        "chunk_size": 32768
    },

    "hardware": {
        "gpus": 1,
        "gpu_type": "H100"
    }
}
```

### Script Wrapper: run_experiment.sh

```bash
#!/bin/bash
# Uso: bash run_experiment.sh configs/xsa_b1_r02.json

CONFIG=$1
RUN_ID=$(python3 -c "import json; print(json.load(open('$CONFIG'))['run_id'])")
LOG_DIR="experiments/logs/${RUN_ID}"
mkdir -p "$LOG_DIR"

echo "=== Starting $RUN_ID ==="
echo "Config: $CONFIG"
echo "Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Estrai parametri dal config
SEED=$(python3 -c "import json; c=json.load(open('$CONFIG')); print(c['training']['seed'])")
QK_GAIN=$(python3 -c "import json; c=json.load(open('$CONFIG')); print(c['training']['qk_gain_init'])")
STEPS=$(python3 -c "import json; c=json.load(open('$CONFIG')); print(c['training']['total_steps'])")
NGPU=$(python3 -c "import json; c=json.load(open('$CONFIG')); print(c['hardware']['gpus'])")
XSA_LAYERS=$(python3 -c "import json; c=json.load(open('$CONFIG')); print(','.join(map(str,c['xsa']['layers'])) if c['xsa']['enabled'] else 'none')")
TTT=$(python3 -c "import json; c=json.load(open('$CONFIG')); print(1 if c['ttt']['enabled'] else 0)")

# Training
SEED=$SEED \
QK_GAIN_INIT=$QK_GAIN \
MAX_STEPS=$STEPS \
XSA_LAYERS=$XSA_LAYERS \
TTT_ENABLED=$TTT \
torchrun --standalone --nproc_per_node=$NGPU train_gpt.py \
    2>&1 | tee "$LOG_DIR/train.log"

# Estrai metriche dal log
python3 analyze_results.py extract "$LOG_DIR/train.log" "$CONFIG" >> experiments/results/all_results.csv

echo "=== Finished $RUN_ID ==="
```

### Script di Analisi: analyze_results.py

```python
#!/usr/bin/env python3
"""
Aggregazione risultati esperimenti Parameter Golf.

Uso:
  python3 analyze_results.py extract <log_file> <config_file>   # Estrai metriche
  python3 analyze_results.py report                              # Report completo
  python3 analyze_results.py compare <baseline_id> <test_id>     # Confronto A/B
"""

import sys
import json
import csv
import re
from pathlib import Path

RESULTS_FILE = "experiments/results/all_results.csv"
FIELDS = [
    "run_id", "block", "xsa_config", "xsa_layers", "qk_gain",
    "depth_rec", "dr_frac", "leaky_slope", "rope_dims",
    "ema_decay", "warmdown", "qat_frac", "ttt_enabled",
    "ttt_lr", "ttt_epochs", "ttt_chunk",
    "steps", "seed", "gpus",
    "loss_500", "loss_1000", "loss_1500", "loss_final",
    "bpb_sliding", "bpb_ttt", "artifact_bytes",
    "train_time_s", "eval_time_s",
    "notes"
]

def extract_metrics(log_path, config_path):
    """Estrai metriche da un log di training e config."""
    with open(config_path) as f:
        cfg = json.load(f)

    with open(log_path) as f:
        log = f.read()

    # Parsing loss a step specifici
    losses = {}
    for match in re.finditer(r'step\s+(\d+).*?val_loss[=:\s]+([\d.]+)', log):
        step = int(match.group(1))
        loss = float(match.group(2))
        losses[step] = loss

    # Parsing BPB
    bpb_sliding = None
    bpb_ttt = None
    for match in re.finditer(r'sliding.*?bpb[=:\s]+([\d.]+)', log, re.IGNORECASE):
        bpb_sliding = float(match.group(1))
    for match in re.finditer(r'ttt.*?bpb[=:\s]+([\d.]+)', log, re.IGNORECASE):
        bpb_ttt = float(match.group(1))

    # Parsing tempo e artifact size
    train_time = None
    for match in re.finditer(r'training.*?(\d+\.?\d*)s', log, re.IGNORECASE):
        train_time = float(match.group(1))
    artifact_bytes = None
    for match in re.finditer(r'artifact.*?(\d+)\s*bytes', log, re.IGNORECASE):
        artifact_bytes = int(match.group(1))

    # Costruisci riga CSV
    xsa_layers_str = ",".join(map(str, cfg["xsa"]["layers"])) if cfg["xsa"]["enabled"] else "none"

    row = {
        "run_id": cfg["run_id"],
        "block": cfg["block"],
        "xsa_config": "custom" if cfg["xsa"]["enabled"] else "none",
        "xsa_layers": xsa_layers_str,
        "qk_gain": cfg["training"]["qk_gain_init"],
        "depth_rec": "3layer" if cfg["depth_recurrence"]["enabled"] else "none",
        "dr_frac": cfg["depth_recurrence"]["activation_frac"],
        "leaky_slope": cfg["architecture"]["leaky_slope"],
        "rope_dims": cfg["architecture"]["rope_dims"],
        "ema_decay": cfg["training"]["ema_decay"],
        "warmdown": cfg["training"]["warmdown_frac"],
        "qat_frac": cfg["training"]["qat_start_frac"],
        "ttt_enabled": cfg["ttt"]["enabled"],
        "ttt_lr": cfg["ttt"]["lr"],
        "ttt_epochs": cfg["ttt"]["epochs"],
        "ttt_chunk": cfg["ttt"]["chunk_size"],
        "steps": cfg["training"]["total_steps"],
        "seed": cfg["training"]["seed"],
        "gpus": cfg["hardware"]["gpus"],
        "loss_500": losses.get(500, ""),
        "loss_1000": losses.get(1000, ""),
        "loss_1500": losses.get(1500, ""),
        "loss_final": losses.get(cfg["training"]["total_steps"], ""),
        "bpb_sliding": bpb_sliding or "",
        "bpb_ttt": bpb_ttt or "",
        "artifact_bytes": artifact_bytes or "",
        "train_time_s": train_time or "",
        "eval_time_s": "",
        "notes": ""
    }

    # Scrivi su CSV
    csv_path = Path(RESULTS_FILE)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Extracted: {cfg['run_id']} | loss@1500={losses.get(1500, 'N/A')}")


def report():
    """Stampa report aggregato ordinato per loss."""
    rows = []
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Ordina per loss_1500 (proxy) o loss_final
    def sort_key(r):
        val = r.get("loss_1500") or r.get("loss_final") or "999"
        try:
            return float(val)
        except ValueError:
            return 999.0

    rows.sort(key=sort_key)

    print(f"\n{'='*100}")
    print(f"{'Run ID':<35} {'XSA':<20} {'QK-Gain':<8} {'Loss@1500':<10} {'BPB-TTT':<10} {'Delta vs BL':<10}")
    print(f"{'='*100}")

    baseline_loss = None
    for r in rows:
        if r["xsa_config"] == "none" and r["block"] == "1":
            baseline_loss = float(r["loss_1500"]) if r["loss_1500"] else None

    for r in rows:
        loss = r.get("loss_1500", "")
        delta = ""
        if baseline_loss and loss:
            try:
                delta = f"{float(loss) - baseline_loss:+.4f}"
            except ValueError:
                pass
        print(f"{r['run_id']:<35} {r['xsa_layers']:<20} {r['qk_gain']:<8} {loss:<10} {r.get('bpb_ttt',''):<10} {delta:<10}")

    print(f"\nTotal experiments: {len(rows)}")
    if baseline_loss:
        print(f"Baseline loss@1500: {baseline_loss:.4f}")


def compare(baseline_id, test_id):
    """Confronto dettagliato tra due run."""
    rows = {}
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["run_id"]] = row

    if baseline_id not in rows or test_id not in rows:
        print(f"Error: run_id not found. Available: {list(rows.keys())}")
        return

    b = rows[baseline_id]
    t = rows[test_id]

    print(f"\n{'Metric':<25} {'Baseline':<20} {'Test':<20} {'Delta':<15}")
    print("-" * 80)

    compare_fields = [
        ("XSA layers", "xsa_layers"),
        ("QK-Gain", "qk_gain"),
        ("Loss @ 500", "loss_500"),
        ("Loss @ 1000", "loss_1000"),
        ("Loss @ 1500", "loss_1500"),
        ("Loss final", "loss_final"),
        ("BPB sliding", "bpb_sliding"),
        ("BPB TTT", "bpb_ttt"),
        ("Artifact bytes", "artifact_bytes"),
        ("Train time (s)", "train_time_s"),
    ]

    for label, field in compare_fields:
        bv = b.get(field, "")
        tv = t.get(field, "")
        delta = ""
        try:
            if bv and tv:
                delta = f"{float(tv) - float(bv):+.4f}"
        except ValueError:
            pass
        print(f"{label:<25} {bv:<20} {tv:<20} {delta:<15}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "extract" and len(sys.argv) == 4:
        extract_metrics(sys.argv[2], sys.argv[3])
    elif cmd == "report":
        report()
    elif cmd == "compare" and len(sys.argv) == 4:
        compare(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
```

### Script Sweep: run_sweep.py

```python
#!/usr/bin/env python3
"""
Genera config e lancia esperimenti in sequenza.

Uso:
  python3 run_sweep.py --block 1              # Lancia Blocco 1
  python3 run_sweep.py --block 2 --best-xsa "3,4,5"  # Blocco 2 con XSA migliore
  python3 run_sweep.py --dry-run --block 1    # Solo genera config senza lanciare
"""

import json
import os
import subprocess
import argparse
from pathlib import Path
from itertools import product

CONFIG_DIR = Path("experiments/configs")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Template di base (record attuale)
BASE_CONFIG = {
    "architecture": {
        "n_layers": 11,
        "d_model": 512,
        "n_heads": 8,
        "n_kv_heads": 4,
        "mlp_expansion": 4,
        "activation": "leaky_relu_sq",
        "leaky_slope": 0.5,
        "rope_dims": 16,
        "logit_softcap": 30.0
    },
    "xsa": {"enabled": False, "layers": []},
    "depth_recurrence": {"enabled": True, "layers": [3,4,5], "activation_frac": 0.35},
    "parallel_residuals": {"enabled": True, "from_layer": 7},
    "training": {
        "qk_gain_init": 5.25,
        "weight_decay": 0.095,
        "max_lr": 0.022,
        "ema_decay": 0.9965,
        "warmdown_frac": 0.72,
        "qat_start_frac": 0.15,
        "total_steps": 1500,  # proxy default
        "seed": 42
    },
    "quantization": {
        "method": "gptq_sdclip",
        "matrix_bits": 6, "matrix_clip_k": 12.85,
        "embedding_bits": 8, "embedding_clip_k": 20.0,
        "compression": "brotli-11"
    },
    "ttt": {"enabled": False, "lr": 0.005, "momentum": 0.9, "epochs": 3, "chunk_size": 32768},
    "hardware": {"gpus": 1, "gpu_type": "H100"}
}

def make_config(run_id, block, overrides, description=""):
    """Crea un config con override specifici."""
    import copy
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["run_id"] = run_id
    cfg["block"] = block
    cfg["description"] = description

    # Applica override nested
    for key_path, value in overrides.items():
        keys = key_path.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value

    path = CONFIG_DIR / f"{run_id}.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def generate_block_1():
    """Blocco 1: Dove applicare XSA."""
    configs = []
    xsa_variants = {
        "baseline_no_xsa": (False, []),
        "xsa_all": (True, list(range(11))),
        "xsa_recurrent": (True, [3, 4, 5]),
        "xsa_non_recurrent": (True, [0,1,2,6,7,8,9,10]),
        "xsa_pre_parallel": (True, list(range(7))),
        "xsa_parallel_only": (True, [7,8,9,10]),
        "xsa_deep": (True, list(range(6, 11))),
        "xsa_shallow": (True, list(range(6))),
    }
    for name, (enabled, layers) in xsa_variants.items():
        run_id = f"b1_{name}"
        overrides = {
            "xsa.enabled": enabled,
            "xsa.layers": layers,
        }
        path = make_config(run_id, 1, overrides, f"Block 1: {name}")
        configs.append(path)
    return configs


def generate_block_2(best_xsa_layers):
    """Blocco 2: QK-Gain sweep."""
    configs = []
    for gain in [3.0, 4.0, 5.25, 6.0, 7.0, 8.0]:
        run_id = f"b2_qkgain_{gain}"
        overrides = {
            "xsa.enabled": True,
            "xsa.layers": best_xsa_layers,
            "training.qk_gain_init": gain,
        }
        path = make_config(run_id, 2, overrides, f"Block 2: QK-Gain {gain}")
        configs.append(path)
    return configs


def generate_block_3(best_xsa_layers):
    """Blocco 3: Depth recurrence variants."""
    configs = []
    dr_variants = [
        ("dr3_frac035", [3,4,5], 0.35),
        ("dr3_frac025", [3,4,5], 0.25),
        ("dr3_frac030", [3,4,5], 0.30),
        ("dr3_frac045", [3,4,5], 0.45),
        ("dr4_frac035", [2,3,4,5], 0.35),
        ("xsa_rec_dr3", [3,4,5], 0.35),  # XSA solo su ricorrenti
    ]
    for name, dr_layers, frac in dr_variants:
        run_id = f"b3_{name}"
        xsa_l = [3,4,5] if "xsa_rec" in name else best_xsa_layers
        overrides = {
            "xsa.enabled": True,
            "xsa.layers": xsa_l,
            "depth_recurrence.layers": dr_layers,
            "depth_recurrence.activation_frac": frac,
        }
        path = make_config(run_id, 3, overrides, f"Block 3: {name}")
        configs.append(path)
    return configs


def generate_block_4(best_xsa_layers):
    """Blocco 4: HP sweep con XSA."""
    configs = []
    sweeps = [
        ("slope03", {"architecture.leaky_slope": 0.3}),
        ("slope04", {"architecture.leaky_slope": 0.4}),
        ("slope06", {"architecture.leaky_slope": 0.6}),
        ("rope24", {"architecture.rope_dims": 24}),
        ("rope32", {"architecture.rope_dims": 32}),
        ("ema0995", {"training.ema_decay": 0.995}),
        ("ema0997", {"training.ema_decay": 0.997}),
        ("ema0998", {"training.ema_decay": 0.998}),
        ("wd065", {"training.warmdown_frac": 0.65}),
        ("wd080", {"training.warmdown_frac": 0.80}),
        ("qat010", {"training.qat_start_frac": 0.10}),
        ("qat020", {"training.qat_start_frac": 0.20}),
    ]
    for name, overrides in sweeps:
        run_id = f"b4_{name}"
        overrides.update({"xsa.enabled": True, "xsa.layers": best_xsa_layers})
        path = make_config(run_id, 4, overrides, f"Block 4: {name}")
        configs.append(path)
    return configs


def run_configs(configs, dry_run=False):
    """Lancia esperimenti in sequenza."""
    for i, config_path in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(configs)}] {config_path.stem}")
        print(f"{'='*60}")
        if dry_run:
            with open(config_path) as f:
                cfg = json.load(f)
            print(f"  XSA: {cfg['xsa']}")
            print(f"  Would run: bash run_experiment.sh {config_path}")
        else:
            subprocess.run(["bash", "run_experiment.sh", str(config_path)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--best-xsa", type=str, default="",
                        help="Comma-separated layer indices, e.g. '3,4,5'")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    best_xsa = [int(x) for x in args.best_xsa.split(",")] if args.best_xsa else list(range(11))

    if args.block == 1:
        configs = generate_block_1()
    elif args.block == 2:
        configs = generate_block_2(best_xsa)
    elif args.block == 3:
        configs = generate_block_3(best_xsa)
    elif args.block == 4:
        configs = generate_block_4(best_xsa)
    else:
        print(f"Block {args.block} non implementato. Usa --block 1-4.")
        exit(1)

    run_configs(configs, dry_run=args.dry_run)
    print(f"\nGenerated {len(configs)} configs in {CONFIG_DIR}/")
```

---

## 6. Setup Ambiente <a name="setup"></a>

### RunPod Setup (1×H100 per proxy, 8×H100 per finale)

```bash
# 1. SSH nel pod RunPod
ssh root@<pod-ip>

# 2. Clone repo
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf

# 3. Dipendenze (già nel template RunPod, ma verifica)
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps \
    --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# 4. Dataset
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
    python3 data/cached_challenge_fineweb.py --variant sp8192

# 5. Crea struttura esperimenti
mkdir -p experiments/{configs,logs,results,analysis}

# 6. Copia i nostri script
# (copia run_experiment.sh, run_sweep.py, analyze_results.py)

# 7. Verifica: riproduci il record (seed 42, 1×H100)
SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=0 \
    torchrun --standalone --nproc_per_node=1 train_gpt.py
```

### Workflow Giornaliero

```bash
# Mattina: lancia blocco proxy
python3 run_sweep.py --block 1 --dry-run          # Verifica config
python3 run_sweep.py --block 1                      # Lancia

# Dopo ~30 min: analizza risultati
python3 analyze_results.py report

# Identifica miglior XSA config
python3 analyze_results.py compare b1_baseline_no_xsa b1_xsa_recurrent

# Pomeriggio: lancia blocco successivo
python3 run_sweep.py --block 2 --best-xsa "3,4,5"
```

---

## 7. Budget e Timeline <a name="budget"></a>

### Costi Stimati

| Fase | Run | GPU | Costo/run | Totale |
|------|-----|-----|-----------|--------|
| Blocco 1: XSA placement | 8 | 1×H100 | $0.15 | $1.20 |
| Blocco 2: QK-Gain | 6 | 1×H100 | $0.15 | $0.90 |
| Blocco 3: Depth recurrence | 6 | 1×H100 | $0.15 | $0.90 |
| Blocco 4: HP sweep | 12 | 1×H100 | $0.15 | $1.80 |
| Blocco 5: TTT (full run) | 4 | 1×H100 | $0.50 | $2.00 |
| Blocco 6: Validazione finale | 6 | 8×H100 | $4.00 | $24.00 |
| Debugging/iterazione | ~20 | mix | ~$0.30 | $6.00 |
| **Totale XSA** | **~62** | | | **~$37** |

### Budget Rimanente per Altre Direzioni

- **Scion optimizer**: ~$50
- **Combinazioni avanzate**: ~$50
- **Riserva debugging**: ~$165
- **Totale disponibile**: $500

### Timeline (15 giorni al 30 aprile)

| Giorno | Attività |
|--------|----------|
| 1 | Setup, riproduci baseline, modifica train_gpt.py con XSA |
| 2 | Blocco 1 + analisi |
| 3 | Blocco 2-3 + analisi |
| 4 | Blocco 4 + analisi |
| 5-6 | Blocco 5 (TTT) + iterazione sulle migliori combo |
| 7 | Blocco 6: validazione 8×H100, 3 seed |
| 8-10 | (Opzionale) Scion exploration |
| 11-12 | Iterazione finale, fix di edge case |
| 13-14 | Scrittura README, preparazione PR |
| 15 | Submit PR |

---

## 8. Criteri di Decisione <a name="decisioni"></a>

### Quando Procedere al Blocco Successivo

- **Blocco 1 → 2**: se almeno una config XSA ha loss@1500 < baseline. Prendi la migliore.
- **Blocco 2 → 3**: se il miglior QK-Gain non è molto diverso dal record (5.25), procedi con 5.25. Se è diverso, usa il nuovo valore.
- **Blocco 3 → 4**: se XSA + depth recurrence non peggiora vs depth recurrence sola.
- **Blocco 4 → 5**: prendi i migliori 2-3 HP che migliorano vs default.
- **Blocco 5 → 6**: se la miglior combo con TTT mostra miglioramento.

### Quando Tagliare le Perdite

- Se Blocco 1 mostra XSA = baseline o peggio su TUTTE le varianti → XSA non funziona in questo regime. Passa a Scion o altre tecniche.
- Se Blocco 3 mostra XSA + depth recurrence peggio di depth recurrence sola → XSA e DR si cannibalizzano. Prova XSA senza DR (meno competitivo ma originale).
- Se dopo Blocco 5 la miglior combo non batte 1.076 BPB → non hai un record, ma puoi fare una PR "non-record" con ablation table interessante.

### Interpretare i Risultati Proxy

La loss@1500 step su 1×H100 **non** è direttamente comparabile al BPB finale su 8×H100 con TTT. Ma:
- I **ranking relativi** tra configurazioni sono affidabili (la config migliore a 1500 step è quasi sempre la migliore anche a 4550)
- I **delta assoluti** si amplificano: un delta di 0.005 a 1500 step → ~0.003-0.008 nel BPB finale
- Il TTT aggiunge un miglioramento quasi costante (~0.002 BPB) indipendentemente dalla config base

---

## 9. Preparazione della PR <a name="pr"></a>

### Struttura del README per la PR

```markdown
# [Titolo]: XSA-{config} + {altre tecniche} ({BPB} BPP, {artifact_size} MB)

## Summary
- Breve descrizione delle novità
- Delta vs SOTA precedente

## Results (3-seed mean)
| Seed | Sliding BPP | TTT BPP | Artifact |
|------|------------|---------|----------|
| 42   | ...        | ...     | ...      |
| 314  | ...        | ...     | ...      |
| 999  | ...        | ...     | ...      |
| Mean | ...        | ...     | ...      |
| Std  | ...        | ...     | ...      |

## Key Techniques
- XSA su layer {X} — spiegazione dell'interazione con depth recurrence
- (Altre modifiche vs record precedente)

## Ablation Table
| Config | Loss@1500 (proxy) | BPB final | Delta |
|--------|-------------------|-----------|-------|
| Baseline (no XSA) | ... | ... | ref |
| XSA-all | ... | ... | ... |
| XSA-recurrent | ... | ... | ... |
| ... | ... | ... | ... |

## Architecture
(Dettagli architettura, come nel record attuale)

## Compliance
(Checklist compliance Track B, identica al record + nota su XSA che non cambia nulla)

## Reproduction
```bash
pip install brotli sentencepiece
SEED=42 QK_GAIN_INIT=... XSA_LAYERS=... TTT_ENABLED=1 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits
- Record precedente: @clarkkev, @dexhunter, etc.
- XSA: Shuangfei Zhai (arXiv:2603.09078)
```

### File da Includere nella PR

1. `README.md` (questo)
2. `submission.json`
3. `train_gpt.py` (con le modifiche XSA)
4. `train_seed42.log`
5. `train_seed314.log`
6. `train_seed999.log`

---

## Note Finali

### Sulla Quantizzazione e XSA

XSA aggiunge due operazioni dopo l'attenzione: `F.normalize(V)` e un dot product + sottrazione.
Queste operano in bf16 durante il training. Dopo la quantizzazione GPTQ:
- I valori di V derivano da pesi int6 → hanno meno precisione
- La normalizzazione amplifica errori di quantizzazione nei valori piccoli
- La proiezione ortogonale potrebbe rimuovere segnale utile se V è "noisy" post-quant

**Mitigazione**: il QAT con STE durante il training dovrebbe adattare il modello a funzionare
con V quantizzati. Ma verifica sempre che l'artifact size non cresca (le nuove operazioni non
aggiungono parametri, solo compute).

### Sul Valore Scientifico

Anche se non batti il record, l'ablation table "XSA × depth recurrence × parallel residuals"
in regime di extreme parameter efficiency è un dato che non esiste nella letteratura.
Il paper XSA dice esplicitamente: "Is [XSA] compatible with other optimizers such as Muon?"
come domanda aperta. I tuoi esperimenti rispondono parzialmente a questa domanda.