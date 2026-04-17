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
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated list of names to run, e.g. 'xsa_recurrent,xsa_deep'")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to use")

    args = parser.parse_args()

    # Update gpus in the base config before generating the blocks
    BASE_CONFIG["hardware"]["gpus"] = args.gpus

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

    # Filter with --only
    if args.only:
        allowed_names = [name.strip() for name in args.only.split(",")]
        # A run_id could be 'b1_xsa_recurrent'. We match if any of the allowed names is in the run_id.
        configs = [c for c in configs if any(a in c.stem for a in allowed_names)]

    run_configs(configs, dry_run=args.dry_run)
    print(f"\nGenerated {len(configs)} configs in {CONFIG_DIR}/")
