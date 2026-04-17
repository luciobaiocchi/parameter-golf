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
ITERATIONS=$STEPS \
XSA_LAYERS=$XSA_LAYERS \
TTT_ENABLED=$TTT \
torchrun --standalone --nproc_per_node=$NGPU train_gpt.py \
    2>&1 | tee "$LOG_DIR/train.log"

# Estrai metriche dal log
python3 analyze_results.py extract "$LOG_DIR/train.log" "$CONFIG" >> experiments/results/all_results.csv

echo "=== Finished $RUN_ID ==="
