#!/bin/bash

# Interrompi l'esecuzione in caso di errore
set -e

VENV_DIR=".venv"

echo "=== Setup Cloud Environment ==="

# 1. Controlla se il venv esiste già, altrimenti lo crea
if [ -d "$VENV_DIR" ]; then
    echo "=> Ambiente virtuale '$VENV_DIR' già esistente."
else
    echo "=> Creazione dell'ambiente virtuale in '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
fi

# Attiva il venv
echo "=> Attivazione dell'ambiente virtuale..."
source "$VENV_DIR/bin/activate"

# 2. Installa le dipendenze
echo "=> Aggiornamento di pip..."
pip install --upgrade pip

echo "=> Installazione delle dipendenze da requirements.txt..."
pip install -r requirements.txt

# 3. Scarica il dataset (usa la cache se già presente)
echo "=> Download del dataset in corso..."
python3 data/cached_challenge_fineweb.py --variant sp1024

echo "=== Setup Completato! ==="
echo "Ora l'ambiente è pronto e il dataset è scaricato/aggiornato."
echo ""
echo "IMPORTANTE: Per attivare l'ambiente nella tua shell corrente, usa il comando:"
echo "source $VENV_DIR/bin/activate"
