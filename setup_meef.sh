#!/bin/bash
# MEEF Setup Script
# Sets up directory structure and builds the parser

echo "╔══════════════════════════════════════════════════════════╗"
echo "║              MEEF Setup Script                           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Create directory structure
echo "[*] Creating directory structure..."
mkdir -p data
mkdir -p data/models
mkdir -p output/ir_results
mkdir -p samples/benign
mkdir -p samples/malicious
mkdir -p samples/dummy

# Create catalog template if it doesn't exist
if [ ! -f data/catalog.csv ]; then
    echo "sha256,label,source,first_seen,local_path,ir_path,notes" > data/catalog.csv
    echo "[✓] Created catalog.csv template"
else
    echo "[✓] catalog.csv already exists"
fi

# Build the parser
echo ""
echo "[*] Building CD Front-End parser..."
cd src/cd_frontend
make clean
make

if [ $? -eq 0 ]; then
    echo ""
    echo "[✓] Parser built successfully!"
    cd ../..
else
    echo ""
    echo "[✗] Parser build failed!"
    cd ../..
    exit 1
fi

# Make meef.py executable
if [ -f meef.py ]; then
    chmod +x meef.py
    echo "[✓] Made meef.py executable"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              Setup Complete!                             ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║ Next steps:                                              ║"
echo "║ 1. Place your .asm samples in samples/ directory        ║"
echo "║ 2. Run: python3 meef.py                                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
