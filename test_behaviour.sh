#!/bin/bash
# Test behavior detection with different samples

echo "╔══════════════════════════════════════════════════════════╗"
echo "║           Behavior Detection Test Suite                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Rebuild parser with fixes
echo "[*] Rebuilding parser with fixes..."
cd src/cd_frontend
make clean > /dev/null 2>&1
make > /dev/null 2>&1
cd ../..

if [ ! -f "src/cd_frontend/meef_parser" ]; then
    echo "[✗] Build failed"
    exit 1
fi

echo "[✓] Parser rebuilt"
echo ""

# Test 1: Create a test file with known API calls
echo "[TEST 1] Non-stripped sample (with API names)"
echo "─────────────────────────────────────────────────"

cat > test_api_sample.asm << 'EOF'
PUSH EBP
MOV EBP, ESP
SUB ESP, 0x10
CALL CreateFileA
TEST EAX, EAX
JZ error_label
CALL WriteFile
CALL InternetConnectA
CALL RegSetValueExA
CALL VirtualAlloc
CALL CreateRemoteThread
XOR EAX, EAX
XOR EBX, EBX
XOR ECX, ECX
error_label:
POP EBP
RET
EOF

./src/cd_frontend/meef_parser test_api_sample.asm output/test_api_ir.json > /dev/null 2>&1

echo "APIs extracted:"
jq '.apis[] | select(.name | test("[A-Z][a-z]+")) | .name' output/test_api_ir.json 2>/dev/null | head -10

echo ""
echo "Behavior detected:"
jq '.behavior' output/test_api_ir.json 2>/dev/null

echo ""
echo "Expected: uses_network=1, uses_fileops=1, uses_registry=1,"
echo "          uses_memory=1, uses_injection=1, uses_crypto=1"
echo ""

# Test 2: Stripped sample (no API names, only addresses)
echo "[TEST 2] Stripped sample (heuristic detection)"
echo "─────────────────────────────────────────────────"

cat > test_stripped_sample.asm << 'EOF'
PUSH RBP
MOV RBP, RSP
SUB RSP, 0x40
MOV RCX, QWORD PTR [RIP+0x1234]
CALL QWORD PTR [RIP+0x5678]
TEST EAX, EAX
JZ label1
XOR R8, R8
XOR R9, R9
XOR R10, R10
XOR R11, R11
CALL QWORD PTR [RIP+0x9ABC]
MOV RDX, QWORD PTR [RIP+0xDEF0]
CALL QWORD PTR [RIP+0x2345]
CMP EAX, 0x0
JNE label2
MOV ECX, 0x1000
CALL QWORD PTR [RIP+0x6789]
label1:
XOR EAX, EAX
label2:
ADD RSP, 0x40
POP RBP
RET
EOF

./src/cd_frontend/meef_parser test_stripped_sample.asm output/test_stripped_ir.json > /dev/null 2>&1

echo "APIs extracted (should be minimal/addresses):"
jq '.apis | length' output/test_stripped_ir.json 2>/dev/null

echo ""
echo "Behavior detected (via heuristics):"
jq '.behavior' output/test_stripped_ir.json 2>/dev/null

echo ""
echo "Expected: Some behavior flags set based on:"
echo "          - Multiple XOR (crypto)"
echo "          - Multiple CALLs (file/memory ops)"
echo "          - High complexity (various behaviors)"
echo ""

# Test 3: Your actual malicious sample
echo "[TEST 3] Your actual malicious sample"
echo "─────────────────────────────────────────────────"

if [ -f "samples/malicious/sus.asm" ]; then
    ./src/cd_frontend/meef_parser samples/malicious/sus.asm output/test_real_malware_ir.json > /dev/null 2>&1
    
    echo "Behavior detected:"
    jq '.behavior' output/test_real_malware_ir.json 2>/dev/null
    
    echo ""
    echo "CFG metrics:"
    jq '.cfg' output/test_real_malware_ir.json 2>/dev/null
    
    echo ""
    
    # Count behavior flags set
    behavior_count=$(jq '[.behavior[] | select(. == 1)] | length' output/test_real_malware_ir.json 2>/dev/null)
    
    if [ "$behavior_count" -gt 0 ]; then
        echo "[✓] SUCCESS: $behavior_count behavior flags detected!"
    else
        echo "[⚠] WARNING: No behavior detected (may need more heuristics)"
    fi
else
    echo "[⚠] Sample not found: samples/malicious/sus.asm"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                      Test Complete                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "1. If Test 1 shows behavior flags → Parser is working"
echo "2. If Test 2 shows behavior flags → Heuristics are working"
echo "3. If Test 3 shows behavior flags → Your malware is detected!"
echo ""
echo "If all tests fail, check:"
echo "  - Parser compiled correctly: make clean && make"
echo "  - semantic_analyzer.c has the new heuristics"
echo "  - Run with: ./test_behavior.sh"
echo ""
