#!/usr/bin/env python3
"""
MEEF Master Orchestrator
Batch processes malware samples through the CD Front-End pipeline
"""

import os
import sys
import subprocess
import json
import hashlib
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

class MEEFOrchestrator:
    def __init__(self):
        self.parser_path = "./src/cd_frontend/meef_parser"
        self.output_dir = "./output/ir_results"
        self.catalog_path = "./data/catalog.csv"
        self.samples_dir = "./samples"
        
    def print_banner(self):
        print(f"\n{Colors.CYAN}{Colors.BOLD}")
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║                    MEEF ORCHESTRATOR                         ║")
        print("║         Malware Executable Exploration Framework             ║")
        print("║              Compiler Design Pipeline Manager                ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print(f"{Colors.RESET}\n")
    
    def calculate_sha256(self, filepath: str) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"{Colors.RED}Error hashing {filepath}: {e}{Colors.RESET}")
            return "HASH_ERROR"
    
    def find_samples(self, directory: str, extensions: List[str] = ['.asm']) -> List[Path]:
        """Find all sample files in directory (only .asm for now)"""
        samples = []
        search_path = Path(directory)
        
        if not search_path.exists():
            print(f"{Colors.RED}✗ Directory not found: {directory}{Colors.RESET}")
            return []
        
        for ext in extensions:
            samples.extend(search_path.rglob(f"*{ext}"))
        
        return sorted(samples)
    
    def display_samples(self, samples: List[Path]) -> None:
        """Display found samples in a nice table"""
        print(f"{Colors.BOLD}Found {len(samples)} sample(s):{Colors.RESET}\n")
        print(f"{Colors.CYAN}{'#':<4} {'Filename':<40} {'Size':<12} {'Type':<10}{Colors.RESET}")
        print("─" * 70)
        
        for idx, sample in enumerate(samples, 1):
            size = sample.stat().st_size
            size_str = self.format_size(size)
            file_type = sample.suffix[1:].upper()
            print(f"{idx:<4} {sample.name:<40} {size_str:<12} {file_type:<10}")
        print()
    
    def format_size(self, size: int) -> str:
        """Format file size human-readable"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} TB"
    
    def get_user_choice(self, samples: List[Path]) -> List[Path]:
        """Let user choose which samples to process"""
        print(f"{Colors.YELLOW}Options:{Colors.RESET}")
        print("  [A] Process ALL samples")
        print("  [#] Process specific sample (enter number)")
        print("  [Q] Quit\n")
        
        choice = input(f"{Colors.BOLD}Your choice: {Colors.RESET}").strip().upper()
        
        if choice == 'Q':
            print(f"{Colors.YELLOW}Exiting...{Colors.RESET}")
            sys.exit(0)
        elif choice == 'A':
            return samples
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(samples):
                return [samples[idx]]
            else:
                print(f"{Colors.RED}Invalid sample number!{Colors.RESET}")
                return []
        else:
            print(f"{Colors.RED}Invalid choice!{Colors.RESET}")
            return []
    
    def run_parser(self, input_file: Path, output_file: Path) -> bool:
        """Run the MEEF parser on a single file"""
        try:
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Run the parser
            cmd = [self.parser_path, str(input_file), str(output_file)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Display output
            if result.stdout:
                print(result.stdout)
            
            if result.returncode != 0:
                print(f"{Colors.RED}✗ Parser failed!{Colors.RESET}")
                if result.stderr:
                    print(f"{Colors.RED}{result.stderr}{Colors.RESET}")
                return False
            
            return True
            
        except Exception as e:
            print(f"{Colors.RED}✗ Error running parser: {e}{Colors.RESET}")
            return False
    
    def update_catalog(self, sample_path: Path, ir_path: Path, label: str = "unknown") -> None:
        """Update catalog.csv with sample metadata"""
        sha256 = self.calculate_sha256(str(sample_path))
        first_seen = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source = "local"
        
        # Determine label from directory structure if possible
        if "malicious" in str(sample_path).lower():
            label = "malicious"
        elif "benign" in str(sample_path).lower():
            label = "benign"
        
        # Read IR to get behavioral flags
        behavior_notes = []
        try:
            with open(ir_path, 'r') as f:
                ir_data = json.load(f)
                behavior = ir_data.get('behavior', {})
                
                if behavior.get('uses_network'): behavior_notes.append("network")
                if behavior.get('uses_fileops'): behavior_notes.append("fileops")
                if behavior.get('uses_injection'): behavior_notes.append("injection")
                if behavior.get('uses_crypto'): behavior_notes.append("crypto")
                
        except Exception as e:
            print(f"{Colors.YELLOW}Warning: Could not read IR for notes: {e}{Colors.RESET}")
        
        notes = ", ".join(behavior_notes) if behavior_notes else "none"
        
        # Create or update catalog
        catalog_dir = Path(self.catalog_path).parent
        catalog_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists and has content
        file_exists = Path(self.catalog_path).exists()
        
        # Read existing entries to avoid duplicates
        existing_hashes = set()
        if file_exists:
            try:
                with open(self.catalog_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_hashes.add(row.get('sha256', ''))
            except Exception:
                pass
        
        # Write header if new file
        if not file_exists or Path(self.catalog_path).stat().st_size == 0:
            with open(self.catalog_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['sha256', 'label', 'source', 'first_seen', 'local_path', 'ir_path', 'notes'])
        
        # Append new entry if hash is new
        if sha256 not in existing_hashes:
            with open(self.catalog_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    sha256,
                    label,
                    source,
                    first_seen,
                    str(sample_path),
                    str(ir_path),
                    notes
                ])
            print(f"{Colors.GREEN}✓ Catalog updated{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}⚠ Sample already in catalog (skipped){Colors.RESET}")
    
    def process_samples(self, samples: List[Path]) -> None:
        """Process multiple samples"""
        total = len(samples)
        successful = 0
        failed = 0
        
        print(f"\n{Colors.BOLD}Processing {total} sample(s)...{Colors.RESET}\n")
        
        for idx, sample in enumerate(samples, 1):
            print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")
            print(f"{Colors.BOLD}[{idx}/{total}] Processing: {sample.name}{Colors.RESET}")
            print(f"{Colors.CYAN}{'='*70}{Colors.RESET}\n")
            
            # Generate output filename
            output_filename = f"{sample.stem}_ir.json"
            output_path = Path(self.output_dir) / output_filename
            
            # Run parser
            success = self.run_parser(sample, output_path)
            
            if success:
                successful += 1
                # Update catalog
                print(f"\n{Colors.BLUE}Updating catalog...{Colors.RESET}")
                self.update_catalog(sample, output_path)
            else:
                failed += 1
            
            print()
        
        # Final summary
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}PROCESSING COMPLETE{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.GREEN}✓ Successful: {successful}{Colors.RESET}")
        if failed > 0:
            print(f"{Colors.RED}✗ Failed: {failed}{Colors.RESET}")
        print(f"\n{Colors.BOLD}Output directory:{Colors.RESET} {self.output_dir}")
        print(f"{Colors.BOLD}Catalog file:{Colors.RESET} {self.catalog_path}\n")
    
    def check_dependencies(self) -> bool:
        """Check if meef_parser exists"""
        if not Path(self.parser_path).exists():
            print(f"{Colors.RED}✗ Parser not found at: {self.parser_path}{Colors.RESET}")
            print(f"{Colors.YELLOW}Please build it first:{Colors.RESET}")
            print(f"  cd src/cd_frontend && make\n")
            return False
        return True
    
    def run(self):
        """Main orchestrator flow"""
        self.print_banner()
        
        # Check dependencies
        if not self.check_dependencies():
            sys.exit(1)
        
        # Get samples directory from user
        default_path = self.samples_dir
        print(f"{Colors.BOLD}Enter path to samples directory{Colors.RESET}")
        print(f"{Colors.YELLOW}(Press Enter for default: {default_path}){Colors.RESET}")
        
        user_path = input(f"{Colors.BOLD}Path: {Colors.RESET}").strip()
        samples_path = user_path if user_path else default_path
        
        # Find samples
        print(f"\n{Colors.BLUE}Scanning {samples_path}...{Colors.RESET}\n")
        samples = self.find_samples(samples_path)
        
        if not samples:
            print(f"{Colors.RED}✗ No samples found!{Colors.RESET}")
            print(f"{Colors.YELLOW}Supported: Only .asm text files (not binary .exe/.dll){Colors.RESET}")
            print(f"{Colors.YELLOW}Tip: Use objdump/IDA/Ghidra to disassemble binaries first{Colors.RESET}\n")
            sys.exit(1)
        
        # Display samples
        self.display_samples(samples)
        
        # Get user choice
        selected = self.get_user_choice(samples)
        
        if not selected:
            print(f"{Colors.RED}No samples selected!{Colors.RESET}\n")
            sys.exit(1)
        
        # Process samples
        self.process_samples(selected)


if __name__ == "__main__":
    orchestrator = MEEFOrchestrator()
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user{Colors.RESET}\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}Fatal error: {e}{Colors.RESET}\n")
        sys.exit(1)
