#!/bin/bash
# Clean disassembly helper for MEEF
# Produces parser-friendly output

print_usage() {
    echo "Usage:"
    echo "  Single file:  ./disassemble_clean.sh <file.exe> [output.asm]"
    echo "  Batch folder: ./disassemble_clean.sh <folder_path>"
    echo ""
    echo "This version cleans objdump output to be parser-friendly"
}

if [ $# -eq 0 ]; then
    print_usage
    exit 1
fi

# Check if objdump is available
if ! command -v objdump &> /dev/null; then
    echo "✗ objdump not found!"
    echo "Install with: sudo apt install binutils"
    exit 1
fi

clean_disassembly() {
    local RAW_FILE="$1"
    local CLEAN_FILE="$2"
    
    # Extract just the instruction mnemonics and operands
    # Remove addresses, hex bytes, section headers, etc.
    grep -E "^\s+[0-9a-f]+:" "$RAW_FILE" | \
        sed -E 's/^\s+[0-9a-f]+:\s+[0-9a-f ]+\s+//' | \
        sed -E 's/\s+#.*//' | \
        sed -E 's/\s+$//' | \
        tr '[:lower:]' '[:upper:]' | \
        grep -v "^$" > "$CLEAN_FILE"
}

disassemble_file() {
    local INPUT="$1"
    local OUTPUT="$2"
    
    # If no output specified, use input name with .asm extension
    if [ -z "$OUTPUT" ]; then
        OUTPUT="${INPUT%.*}.asm"
    fi
    
    # Check if input file exists
    if [ ! -f "$INPUT" ]; then
        echo "✗ File not found: $INPUT"
        return 1
    fi
    
    echo "[*] Disassembling: $(basename "$INPUT")"
    
    # Create temporary raw disassembly
    local TEMP_RAW="${OUTPUT}.tmp"
    
    # Disassemble with Intel syntax
    objdump -d -M intel "$INPUT" > "$TEMP_RAW" 2>/dev/null
    
    if [ $? -ne 0 ]; then
        echo "[✗] objdump failed"
        rm -f "$TEMP_RAW"
        return 1
    fi
    
    # Clean the output
    clean_disassembly "$TEMP_RAW" "$OUTPUT"
    rm -f "$TEMP_RAW"
    
    # Verify output
    if [ ! -f "$OUTPUT" ] || [ ! -s "$OUTPUT" ]; then
        echo "[✗] Cleaning produced empty output"
        return 1
    fi
    
    LINES=$(wc -l < "$OUTPUT" 2>/dev/null)
    SIZE=$(stat -c%s "$OUTPUT" 2>/dev/null || stat -f%z "$OUTPUT" 2>/dev/null)
    
    if [ "$SIZE" -gt 100 ]; then
        echo "[✓] Success! ($LINES instructions, $(numfmt --to=iec-i --suffix=B $SIZE 2>/dev/null || echo $SIZE bytes))"
        
        # Show preview
        echo ""
        echo "Preview (first 10 lines):"
        head -10 "$OUTPUT"
        echo ""
        
        return 0
    else
        echo "[✗] Output too small (likely failed)"
        rm -f "$OUTPUT"
        return 1
    fi
}

# Main logic
INPUT_PATH="$1"
OUTPUT_PATH="$2"

# Check if input is a directory
if [ -d "$INPUT_PATH" ]; then
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║           Batch Clean Disassembly Mode                  ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    echo "[*] Scanning folder: $INPUT_PATH"
    
    # Find all .exe and .dll files
    FILES=$(find "$INPUT_PATH" -maxdepth 1 -type f \( -iname "*.exe" -o -iname "*.dll" \))
    COUNT=$(echo "$FILES" | grep -c . 2>/dev/null)
    
    if [ -z "$FILES" ] || [ "$COUNT" -eq 0 ]; then
        echo "[✗] No .exe or .dll files found in $INPUT_PATH"
        exit 1
    fi
    
    echo "[✓] Found $COUNT file(s)"
    echo ""
    
    # Process each file
    SUCCESS=0
    FAILED=0
    
    while IFS= read -r FILE; do
        if [ -n "$FILE" ]; then
            if disassemble_file "$FILE"; then
                ((SUCCESS++))
            else
                ((FAILED++))
            fi
            echo ""
        fi
    done <<< "$FILES"
    
    # Summary
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║                   Disassembly Complete                   ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo "║ Total files:     $COUNT"
    echo "║ Success:         $SUCCESS"
    echo "║ Failed:          $FAILED"
    echo "╚══════════════════════════════════════════════════════════╝"
    
elif [ -f "$INPUT_PATH" ]; then
    # Single file mode
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║          Clean Single File Disassembly                   ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    
    if disassemble_file "$INPUT_PATH" "$OUTPUT_PATH"; then
        OUTPUT="${OUTPUT_PATH:-${INPUT_PATH%.*}.asm}"
        echo "Ready to parse:"
        echo "  ./src/cd_frontend/meef_parser $OUTPUT output/result_ir.json"
        exit 0
    else
        exit 1
    fi
    
else
    echo "✗ Input not found: $INPUT_PATH"
    echo ""
    print_usage
    exit 1
fi
