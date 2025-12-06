# 🔧 MEEF Troubleshooting Guide

## Parse Errors - "syntax error at line X"

### Issue 1: Binary Files (`.exe`, `.dll`)
**Error:**
```
Parse error at line 1: syntax error
[✗] Parsing failed
```

**Cause:** The parser only handles TEXT assembly files (`.asm`), not binary executables.

**Solution:** Disassemble binary files first:

```bash
# Option 1: Use objdump (Linux/WSL)
objdump -d -M intel calc.exe > calc.asm

# Option 2: Use the helper script
chmod +x disassemble.sh
./disassemble.sh calc.exe calc.asm

# Option 3: Use other tools
# - IDA Pro (commercial)
# - Ghidra (free)
# - radare2 (free)
# - Binary Ninja (commercial)
```

Then process the `.asm` file:
```bash
python3 meef.py
# Select calc.asm instead of calc.exe
```

---

### Issue 2: Complex Assembly Syntax
**Error:**
```
Parse error at line 15: syntax error
Parse error at line 23: syntax error
...
```

**Cause:** The assembly file contains syntax our simple parser doesn't recognize:
- Assembler directives (`.text`, `.data`, `.section`)
- Complex addressing modes
- Macro definitions
- Non-standard instruction formats

**Solutions:**

**A) Clean the ASM file:**
```bash
# Remove directives and comments
grep -v "^\." calc.asm | grep -v "^#" | grep -v "^;" > calc_clean.asm

# Try parsing the cleaned version
./src/cd_frontend/meef_parser calc_clean.asm output/calc_ir.json
```

**B) Use validate_asm.py to check first:**
```bash
python3 validate_asm.py samples/dummy/calc.asm
```

This shows you what the file looks like and if it's parseable.

**C) Extract only instructions:**
```bash
# Keep only lines with common opcodes
grep -E "(MOV|CALL|JMP|PUSH|POP|RET|ADD|SUB|XOR)" calc.asm > calc_simplified.asm
```

---

### Issue 3: Wrong File Format
**Error:**
```
Parse error at line 2: syntax error
```

**Cause:** File might be:
- AT&T syntax instead of Intel syntax
- NASM/MASM specific format
- Disassembly with extra metadata

**Check the format:**
```bash
head -20 samples/dummy/calc.asm
```

**Intel syntax looks like:**
```asm
MOV eax, ebx
CALL CreateFileA
PUSH ebp
```

**AT&T syntax looks like (wrong for us):**
```asm
movl %ebx, %eax
call CreateFileA
pushq %rbp
```

**Fix:** Convert AT&T to Intel:
```bash
# If you have AT&T syntax from objdump
objdump -d -M intel yourfile.exe > yourfile.asm
```

---

## No Samples Found

**Error:**
```
✗ No samples found!
```

**Fixes:**

1. **Check directory:**
```bash
ls -la samples/dummy/
```

2. **Verify file extensions:**
```bash
# Only .asm files are scanned now (not .exe or .dll)
find samples/ -name "*.asm"
```

3. **Create test sample:**
```bash
cat > samples/dummy/test.asm << 'EOF'
CALL CreateFileA
MOV eax, ebx
RET
EOF
```

---

## Parser Not Found

**Error:**
```
✗ Parser not found at: ./src/cd_frontend/meef_parser
```

**Fix:**
```bash
cd src/cd_frontend
make clean
make
cd ../..
```

---

## Build Errors

### Missing Dependencies
**Error:**
```
bison: command not found
flex: command not found
```

**Fix:**
```bash
# Ubuntu/Debian
sudo apt install gcc flex bison make

# Fedora/RHEL
sudo dnf install gcc flex bison make
```

### Multiple Definition Errors
**Error:**
```
multiple definition of `ctx_init'
```

**Fix:** You're using old files. Update to the new structure:
- Ensure `cd_context.c` exists
- Check `Makefile` includes `cd_context.c` in SOURCES

---

## Partial Success (Some Files Fail)

**Scenario:** Processing 10 files, 7 succeed, 3 fail.

**This is NORMAL!** Some assembly files are just too complex or malformed.

**What to do:**
1. Check which files failed:
```bash
ls output/ir_results/
# Compare with samples/ to see what's missing
```

2. Manually validate failed files:
```bash
python3 validate_asm.py samples/malicious/complex_malware.asm
```

3. Simplify or skip problematic files

---

## Catalog Not Updating

**Issue:** `catalog.csv` is empty or missing entries.

**Fixes:**

1. **Check permissions:**
```bash
ls -la data/catalog.csv
chmod 664 data/catalog.csv
```

2. **Manually create header:**
```bash
echo "sha256,label,source,first_seen,local_path,ir_path,notes" > data/catalog.csv
```

3. **Check for errors in meef.py output:**
Look for "Catalog updated" messages

---

## Performance Issues

**Issue:** Processing is very slow.

**Solutions:**

1. **Process in batches:**
```bash
# Process benign first
python3 meef.py
Path: samples/benign
Choice: A

# Then malicious
python3 meef.py
Path: samples/malicious
Choice: A
```

2. **Skip large files:**
```bash
# Find large files
find samples/ -name "*.asm" -size +1M

# Move them aside
mkdir samples/large_files
mv samples/*/huge_file.asm samples/large_files/
```

---

## Quick Diagnostic Checklist

Run these commands to check your setup:

```bash
# 1. Check parser exists
ls -la src/cd_frontend/meef_parser

# 2. Check samples exist
find samples/ -name "*.asm"

# 3. Test parser manually
./src/cd_frontend/meef_parser samples/dummy/fake.asm output/test.json

# 4. Check output
cat output/test.json

# 5. Verify catalog
cat data/catalog.csv
```

---

## Still Having Issues?

### Get detailed error info:
```bash
# Run parser with verbose output
./src/cd_frontend/meef_parser yourfile.asm output/debug.json 2>&1 | tee parser_debug.log
```

### Check sample file format:
```bash
# View file type
file samples/dummy/calc.exe

# View first 20 lines
head -20 samples/dummy/calc.asm

# Check for binary data
python3 validate_asm.py samples/dummy/calc.asm
```

### Simplify test case:
```bash
# Create minimal test
cat > test_minimal.asm << 'EOF'
CALL CreateFileA
MOV eax, ebx
RET
EOF

# Try parsing
./src/cd_frontend/meef_parser test_minimal.asm output/minimal.json
```

---

## Common File Preparation Workflow

```bash
# 1. Download/collect malware samples (.exe files)
# 2. Disassemble them
for exe in samples/malicious/*.exe; do
    asm="${exe%.exe}.asm"
    objdump -d -M intel "$exe" > "$asm"
done

# 3. Validate
for asm in samples/malicious/*.asm; do
    python3 validate_asm.py "$asm"
done

# 4. Process with MEEF
python3 meef.py
```

---

## Best Practices

✅ **DO:**
- Use Intel syntax assembly
- Clean/preprocess complex ASM files
- Validate files before batch processing
- Process in small batches first
- Keep original files separate from disassembled versions

❌ **DON'T:**
- Try to parse binary .exe/.dll files directly
- Process unsupported formats without conversion
- Ignore validation warnings
- Delete original files after disassembly

---

**Need more help? Check the error messages carefully - they usually point to the exact issue!** 🔍
