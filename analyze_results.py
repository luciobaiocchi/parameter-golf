#!/usr/bin/env python3
"""
Aggregazione risultati esperimenti Parameter Golf.

FIXED: regex aggiornati per il formato reale dei log di train_gpt.py.

Uso:
  python3 analyze_results.py extract <log_file> <config_file>   # Estrai metriche
  python3 analyze_results.py report                              # Report completo
  python3 analyze_results.py compare <baseline_id> <test_id>     # Confronto A/B
  python3 analyze_results.py test <log_file>                     # Test regex su un log
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
    "loss_100", "loss_500", "loss_1000", "loss_1500", "loss_final",
    "bpb_final_unquantized", "bpb_final_int8_zlib",
    "artifact_bytes", "artifact_bytes_int8_zlib",
    "train_time_s", "eval_time_s",
    "stopped_early", "last_step_reached",
    "notes"
]


def parse_log(log_text):
    """
    Parsa un log di train_gpt.py e ritorna un dict di metriche.

    Formato atteso:
      step:236/1500 val_loss:3.1865 val_bpb:1.8872 train_time:602006ms
      step:200/1500 train_loss:3.2004 train_time:510181ms
      stopping_early: wallclock_cap train_time:602006ms step:236/1500
      Total submission size: 67273407 bytes
      Total submission size int8+zlib: 8002952 bytes
      final_int8_zlib_roundtrip val_loss:3.2686 val_bpb:1.9358 eval_time:88306ms
    """
    metrics = {}

    # val_loss per ogni step in cui è presente — "step:236/1500 val_loss:3.1865 val_bpb:1.8872"
    val_losses = {}
    val_bpbs = {}
    pattern_val = re.compile(
        r'step:(\d+)/\d+\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)'
    )
    for match in pattern_val.finditer(log_text):
        step = int(match.group(1))
        val_losses[step] = float(match.group(2))
        val_bpbs[step] = float(match.group(3))

    metrics["val_losses"] = val_losses
    metrics["val_bpbs"] = val_bpbs

    # train_loss per ogni step — "step:5/1500 train_loss:9.6426 train_time:12734ms"
    train_losses = {}
    pattern_train = re.compile(
        r'step:(\d+)/\d+\s+train_loss:([\d.]+)\s+train_time:(\d+)ms'
    )
    train_times_by_step = {}
    for match in pattern_train.finditer(log_text):
        step = int(match.group(1))
        train_losses[step] = float(match.group(2))
        train_times_by_step[step] = int(match.group(3))

    metrics["train_losses"] = train_losses
    metrics["train_times_by_step"] = train_times_by_step

    # Ultimo step raggiunto
    last_step = max(
        list(val_losses.keys()) + list(train_losses.keys()) + [0]
    )
    metrics["last_step_reached"] = last_step

    # Stopped early? — "stopping_early: wallclock_cap train_time:602006ms step:236/1500"
    stopped_early_match = re.search(
        r'stopping_early:\s*(\w+)\s+train_time:(\d+)ms\s+step:(\d+)/(\d+)',
        log_text
    )
    if stopped_early_match:
        metrics["stopped_early"] = stopped_early_match.group(1)
        metrics["train_time_ms_final"] = int(stopped_early_match.group(2))
        metrics["last_step_reached"] = int(stopped_early_match.group(3))
    else:
        metrics["stopped_early"] = ""

    # Submission size — "Total submission size: 67273407 bytes"
    size_match = re.search(r'Total submission size:\s*(\d+)\s+bytes', log_text)
    if size_match:
        metrics["artifact_bytes"] = int(size_match.group(1))

    # Submission size int8+zlib — "Total submission size int8+zlib: 8002952 bytes"
    size_int8_match = re.search(
        r'Total submission size int8\+zlib:\s*(\d+)\s+bytes', log_text
    )
    if size_int8_match:
        metrics["artifact_bytes_int8_zlib"] = int(size_int8_match.group(1))

    # BPB finale dopo quantizzazione — "final_int8_zlib_roundtrip_exact val_loss:3.26859778 val_bpb:1.93584772"
    final_quant_match = re.search(
        r'final_int8_zlib_roundtrip(?:_exact)?\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)',
        log_text
    )
    if final_quant_match:
        metrics["final_val_loss_quantized"] = float(final_quant_match.group(1))
        metrics["final_val_bpb_quantized"] = float(final_quant_match.group(2))

    # Eval time — "eval_time:88306ms"
    eval_time_match = re.search(r'eval_time:(\d+)ms', log_text)
    if eval_time_match:
        metrics["eval_time_ms"] = int(eval_time_match.group(1))

    # Train time totale (prendi l'ultimo train_time visto)
    all_train_times = [
        int(m.group(1))
        for m in re.finditer(r'train_time:(\d+)ms', log_text)
    ]
    if all_train_times:
        metrics["train_time_ms_final"] = max(all_train_times)

    # Model params — "model_params:17059912"
    params_match = re.search(r'model_params:(\d+)', log_text)
    if params_match:
        metrics["model_params"] = int(params_match.group(1))

    return metrics


def closest_step_value(step_dict, target_step):
    """Ritorna il valore allo step più vicino a target_step, o '' se non trovato."""
    if not step_dict:
        return ""
    # Cerca esattamente, altrimenti il più vicino ma <= target
    if target_step in step_dict:
        return step_dict[target_step]
    # Trova lo step <= target più grande
    valid_steps = [s for s in step_dict.keys() if s <= target_step]
    if valid_steps:
        return step_dict[max(valid_steps)]
    return ""


def extract_metrics(log_path, config_path):
    """Estrai metriche da un log di training e config."""
    with open(config_path) as f:
        cfg = json.load(f)

    with open(log_path) as f:
        log = f.read()

    metrics = parse_log(log)
    val_losses = metrics.get("val_losses", {})
    train_losses = metrics.get("train_losses", {})

    # Per il proxy, il valore a 1500 step potrebbe non esistere se è stato fermato prima.
    # Prendi il valore più vicino o l'ultimo disponibile.
    target_steps = [100, 500, 1000, 1500]
    loss_at_step = {}
    for target in target_steps:
        # Preferisci val_loss, fallback su train_loss
        v = closest_step_value(val_losses, target)
        if v == "":
            v = closest_step_value(train_losses, target)
        loss_at_step[target] = v

    # Loss finale = ultimo val_loss misurato
    last_step = metrics.get("last_step_reached", 0)
    loss_final = ""
    if val_losses:
        loss_final = val_losses[max(val_losses.keys())]
    elif train_losses:
        loss_final = train_losses[max(train_losses.keys())]

    # Costruisci riga CSV
    xsa_layers_str = (
        ",".join(map(str, cfg["xsa"]["layers"]))
        if cfg["xsa"]["enabled"] else "none"
    )

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
        "loss_100": loss_at_step[100],
        "loss_500": loss_at_step[500],
        "loss_1000": loss_at_step[1000],
        "loss_1500": loss_at_step[1500],
        "loss_final": loss_final,
        "bpb_final_unquantized": closest_step_value(
            metrics.get("val_bpbs", {}), last_step
        ),
        "bpb_final_int8_zlib": metrics.get("final_val_bpb_quantized", ""),
        "artifact_bytes": metrics.get("artifact_bytes", ""),
        "artifact_bytes_int8_zlib": metrics.get("artifact_bytes_int8_zlib", ""),
        "train_time_s": (
            metrics.get("train_time_ms_final", 0) / 1000
            if metrics.get("train_time_ms_final") else ""
        ),
        "eval_time_s": (
            metrics.get("eval_time_ms", 0) / 1000
            if metrics.get("eval_time_ms") else ""
        ),
        "stopped_early": metrics.get("stopped_early", ""),
        "last_step_reached": metrics.get("last_step_reached", 0),
        "notes": ""
    }

    # Scrivi su CSV
    csv_path = Path(RESULTS_FILE)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # Stampa riassunto utile
    last = metrics.get("last_step_reached", 0)
    stopped = metrics.get("stopped_early", "")
    stop_info = f" (STOPPED: {stopped})" if stopped else ""
    bpb_q = row["bpb_final_int8_zlib"]
    bpb_str = f"bpb_int8={bpb_q:.4f}" if isinstance(bpb_q, (int, float)) else "bpb_int8=N/A"
    loss_str = (
        f"loss@{last}={loss_final:.4f}"
        if isinstance(loss_final, (int, float)) else "loss=N/A"
    )
    print(f"Extracted: {cfg['run_id']:<30} {loss_str} | {bpb_str}{stop_info}")


def test_parser(log_path):
    """Testa il parser su un log e stampa tutto ciò che ha trovato."""
    with open(log_path) as f:
        log = f.read()

    metrics = parse_log(log)

    print(f"\n=== Parser test on {log_path} ===\n")
    print(f"Model params: {metrics.get('model_params', 'N/A')}")
    print(f"Last step reached: {metrics.get('last_step_reached', 0)}")
    print(f"Stopped early: {metrics.get('stopped_early', 'no')}")
    print(f"Train time: {metrics.get('train_time_ms_final', 0)/1000:.1f}s")
    print(f"Eval time: {metrics.get('eval_time_ms', 0)/1000:.1f}s")
    print(f"Artifact bytes (raw): {metrics.get('artifact_bytes', 'N/A')}")
    print(f"Artifact bytes (int8+zlib): {metrics.get('artifact_bytes_int8_zlib', 'N/A')}")
    print(f"Final val_bpb quantized: {metrics.get('final_val_bpb_quantized', 'N/A')}")
    print(f"\nVal losses by step:")
    for step, loss in sorted(metrics.get("val_losses", {}).items()):
        bpb = metrics.get("val_bpbs", {}).get(step, "?")
        print(f"  step {step}: val_loss={loss:.4f}, val_bpb={bpb:.4f}" if isinstance(bpb, float) else f"  step {step}: val_loss={loss:.4f}")
    print(f"\nTrain losses (first 5 and last 5):")
    train = sorted(metrics.get("train_losses", {}).items())
    for step, loss in train[:5]:
        print(f"  step {step}: train_loss={loss:.4f}")
    if len(train) > 10:
        print(f"  ...")
        for step, loss in train[-5:]:
            print(f"  step {step}: train_loss={loss:.4f}")


def report():
    """Stampa report aggregato ordinato per BPB finale."""
    if not Path(RESULTS_FILE).exists():
        print(f"Nessun risultato trovato in {RESULTS_FILE}")
        return

    rows = []
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Ordina per BPB int8+zlib (la metrica che conta), fallback su loss_final
    def sort_key(r):
        for field in ["bpb_final_int8_zlib", "loss_final", "loss_1500", "loss_1000"]:
            val = r.get(field, "")
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
        return 999.0

    rows.sort(key=sort_key)

    # Trova baseline per calcolare delta
    baseline_bpb = None
    for r in rows:
        if r["xsa_config"] == "none" and str(r["block"]) == "1":
            try:
                baseline_bpb = float(r["bpb_final_int8_zlib"])
                break
            except (ValueError, TypeError):
                pass

    print(f"\n{'='*120}")
    print(f"{'Run ID':<30} {'XSA Layers':<22} {'QK':<6} {'Last Step':<10} {'Loss Final':<12} {'BPB int8':<10} {'Delta BL':<10} {'Stopped':<12}")
    print(f"{'='*120}")

    for r in rows:
        loss_final = r.get("loss_final", "")
        bpb = r.get("bpb_final_int8_zlib", "")
        try:
            bpb_f = float(bpb)
            bpb_str = f"{bpb_f:.4f}"
            delta = f"{bpb_f - baseline_bpb:+.4f}" if baseline_bpb else ""
        except (ValueError, TypeError):
            bpb_str = ""
            delta = ""
        try:
            loss_str = f"{float(loss_final):.4f}"
        except (ValueError, TypeError):
            loss_str = ""

        last_step = r.get("last_step_reached", "0")
        stopped = r.get("stopped_early", "")

        print(
            f"{r['run_id']:<30} "
            f"{r['xsa_layers']:<22} "
            f"{r['qk_gain']:<6} "
            f"{last_step:<10} "
            f"{loss_str:<12} "
            f"{bpb_str:<10} "
            f"{delta:<10} "
            f"{stopped:<12}"
        )

    print(f"\nTotal experiments: {len(rows)}")
    if baseline_bpb:
        print(f"Baseline BPB (int8+zlib): {baseline_bpb:.4f}")


def compare(baseline_id, test_id):
    """Confronto dettagliato tra due run."""
    rows = {}
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["run_id"]] = row

    if baseline_id not in rows or test_id not in rows:
        print(f"Error: run_id not found.")
        print(f"Available: {sorted(rows.keys())}")
        return

    b = rows[baseline_id]
    t = rows[test_id]

    print(f"\n{'Metric':<28} {'Baseline':<20} {'Test':<20} {'Delta':<15}")
    print("-" * 85)

    compare_fields = [
        ("XSA layers", "xsa_layers"),
        ("QK-Gain", "qk_gain"),
        ("Last step reached", "last_step_reached"),
        ("Loss @ 100", "loss_100"),
        ("Loss @ 500", "loss_500"),
        ("Loss @ 1000", "loss_1000"),
        ("Loss @ 1500", "loss_1500"),
        ("Loss final", "loss_final"),
        ("BPB unquantized", "bpb_final_unquantized"),
        ("BPB int8+zlib", "bpb_final_int8_zlib"),
        ("Artifact bytes (raw)", "artifact_bytes"),
        ("Artifact int8+zlib", "artifact_bytes_int8_zlib"),
        ("Train time (s)", "train_time_s"),
        ("Stopped early", "stopped_early"),
    ]

    for label, field in compare_fields:
        bv = b.get(field, "")
        tv = t.get(field, "")
        delta = ""
        try:
            if bv != "" and tv != "":
                delta = f"{float(tv) - float(bv):+.4f}"
        except (ValueError, TypeError):
            pass
        print(f"{label:<28} {str(bv):<20} {str(tv):<20} {delta:<15}")


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
    elif cmd == "test" and len(sys.argv) == 3:
        test_parser(sys.argv[2])
    else:
        print(__doc__)