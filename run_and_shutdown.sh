#!/bin/bash

echo "Avvio dello sweep per il Blocco 1..."
./run_sweep.py --block 1

# Cattura l'esito dello script
if [ $? -eq 0 ]; then
    echo "================================================="
    echo "Sweep completato con successo!"
else
    echo "================================================="
    echo "Lo sweep ha terminato con un errore o è stato interrotto."
fi

# Lascia un margine per dare un'occhiata prima di spegnere
echo "Spegnimento automatico del sistema in 60 secondi..."
echo "Premi Ctrl+C ora se vuoi annullare lo spegnimento!"
echo "================================================="

sleep 60
systemctl poweroff
