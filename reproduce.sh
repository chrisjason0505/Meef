#!/bin/bash
# Recreate MEeF analysis environment

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "[+] Environment ready."
