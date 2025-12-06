#!/usr/bin/env python3
"""
ASM File Validator - Preview and validate assembly files
"""

import sys
from pathlib import Path

def is_binary(filepath):
    """Check if file is binary"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            f.read(1024)
        return False
    except UnicodeDecodeError:
        return True

def preview_asm(filepath, lines=20):
    """Preview first N lines of ASM file"""
    print(f"\n{'='*70}")
    print(f"File: {filepath}")
    print(f"{'='*70}\n")
    
    if is_binary(filepath):
        print("⚠️  WARNING: This is a BINARY file!")
        print("   Use a disassembler to convert to .asm first")
        print("   Examples: objdump, IDA Pro, Ghidra, radare2")
        return False
    
    try:
        with open(filepath, 'r') as f:
            file_lines = f.readlines()
            total = len(file_lines)
            
            print(f"Total lines: {total}")
            print(f"Preview (first {min(lines, total)} lines):\n")
            
            for i, line in enumerate(file_lines[:lines], 1):
                print(f"{i:4d} | {line.rstrip()}")
            
            if total > lines:
                print(f"\n... ({total - lines} more lines)")
            
            # Quick analysis
            opcodes = ['MOV', 'CALL', 'JMP', 'PUSH', 'POP', 'RET']
            opcode_count = sum(any(op in line.upper() for op in opcodes) for line in file_lines)
            
            print(f"\n📊 Quick Stats:")
            print(f"   Instructions found: ~{opcode_count}")
            print(f"   Looks parseable: {'✓ Yes' if opcode_count > 0 else '✗ No'}")
            
            return opcode_count > 0
            
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 validate_asm.py <file.asm>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    if not Path(filepath).exists():
        print(f"✗ File not found: {filepath}")
        sys.exit(1)
    
    valid = preview_asm(filepath)
    sys.exit(0 if valid else 1)
