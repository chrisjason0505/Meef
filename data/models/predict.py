#!/usr/bin/env python3
"""
Predict if a sample is malware using trained ML model
Supports Windows (WSL) and Linux
"""

import sys
import json
import numpy as np
import joblib
from pathlib import Path
import subprocess
import os
import shlex

class MalwarePredictor:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = Path(__file__).parent
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.metadata = None
        
    def _windows_to_wsl(self, windows_path):
        """Convert Windows path to WSL path"""
        p = str(Path(windows_path).resolve())
        if len(p) >= 2 and p[1] == ':':
            drive = p[0].lower()
            rest = p[2:].replace('\\', '/')
            return f"/mnt/{drive}{rest}"
        return p.replace('\\', '/')
    
    def _run_cmd(self, args, timeout=60):
        """Run command - use WSL on Windows"""
        if os.name == 'nt':
            cmd_str = ' '.join(shlex.quote(str(a)) for a in args)
            return subprocess.run(
                ['wsl', 'bash', '-c', cmd_str],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout
            )
        else:
            return subprocess.run(
                [str(a) for a in args],
                capture_output=True,
                text=True,
                timeout=timeout
            )
        
    def load_model(self):
        """Load trained model"""
        try:
            model_path = self.model_dir / "malware_classifier.pkl"
            scaler_path = self.model_dir / "feature_scaler.pkl"
            metadata_path = self.model_dir / "model_metadata.json"
            
            self.model = joblib.load(model_path)
            
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            else:
                self.scaler = None
            
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Get accuracy from metadata (handle different formats)
            acc = 0
            if 'test_metrics' in self.metadata:
                acc = self.metadata['test_metrics'].get('accuracy', 0)
            elif 'metrics' in self.metadata:
                metrics = self.metadata['metrics']
                if isinstance(metrics, dict):
                    if 'test' in metrics:
                        acc = metrics['test'].get('accuracy', 0)
                    elif 'accuracy' in metrics:
                        acc = metrics['accuracy']
            
            print(f"[✓] Model loaded successfully")
            print(f"    Accuracy: {acc*100:.2f}%")
            return True
            
        except Exception as e:
            print(f"[✗] Error loading model: {e}")
            return False
    
    def process_sample(self, input_path):
        """Process a sample through the pipeline"""
        input_path = Path(input_path)
        
        # Check if it's already an ASM file
        if input_path.suffix == '.asm':
            asm_file = input_path
        elif input_path.suffix in ['.exe', '.dll']:
            # Need to disassemble first
            print(f"[*] Disassembling {input_path.name}...")
            asm_file = input_path.with_suffix('.asm')
            
            # Find disassemble script
            script_dir = Path(__file__).parent.parent.parent
            disasm_script = script_dir / 'disassemble.sh'
            
            if not disasm_script.exists():
                print(f"[✗] Disassembly script not found")
                return None
            
            try:
                if os.name == 'nt':
                    w_script = self._windows_to_wsl(disasm_script)
                    w_input = self._windows_to_wsl(input_path)
                    w_output = self._windows_to_wsl(asm_file)
                    # Fix line endings and run script
                    cmd_str = f"sed 's/\\r$//' '{w_script}' | bash -s '{w_input}' '{w_output}'"
                    result = subprocess.run(
                        ['wsl', 'bash', '-c', cmd_str],
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        timeout=120
                    )
                else:
                    result = self._run_cmd([str(disasm_script), str(input_path), str(asm_file)], timeout=120)
                
                if result.returncode != 0:
                    print(f"[✗] Disassembly failed: {result.stderr or result.stdout}")
                    return None
            except Exception as e:
                print(f"[✗] Disassembly failed: {e}")
                return None
        else:
            print(f"[✗] Unsupported file type: {input_path.suffix}")
            return None
        
        # Parse with MEEF
        print(f"[*] Parsing with MEEF compiler...")
        import uuid
        ir_file = Path(f'output/temp_ir_{uuid.uuid4().hex}.json')
        ir_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure clean state
        if ir_file.exists():
            ir_file.unlink()
        
        # Find parser
        script_dir = Path(__file__).parent.parent.parent
        parser_bin = script_dir / 'src' / 'cd_frontend' / 'meef_parser'
        
        if not parser_bin.exists():
            print(f"[✗] Parser not found: {parser_bin}")
            return None
        
        try:
            if os.name == 'nt':
                w_parser = self._windows_to_wsl(parser_bin)
                w_asm = self._windows_to_wsl(asm_file)
                w_ir = self._windows_to_wsl(ir_file)
                result = self._run_cmd([w_parser, w_asm, w_ir], timeout=60)
            else:
                result = self._run_cmd([str(parser_bin), str(asm_file), str(ir_file)], timeout=60)
            
            if result.returncode != 0 or not ir_file.exists():
                print(f"[✗] Parsing failed: {result.stderr or result.stdout}")
                return None
        except Exception as e:
            print(f"[✗] Parsing failed: {e}")
            return None
        
        # Verify IR file was created and has content
        if not ir_file.exists():
            print(f"[✗] IR file was not created: {ir_file}")
            return None
        
        if ir_file.stat().st_size == 0:
            print(f"[✗] IR file is empty: {ir_file}")
            return None
            
        print(f"[✓] IR generated successfully: {ir_file.name}")
        return ir_file
    
    def extract_features(self, ir_path):
        """Extract features from IR file"""
        try:
            with open(ir_path, 'r') as f:
                ir = json.load(f)
            
            features = []
            feature_names = self.metadata.get('feature_names', self.metadata.get('features', []))
            
            for feat_name in feature_names:
                if feat_name.startswith('uses_'):
                    behavior_key = feat_name.replace('uses_', '')
                    features.append(ir.get('behavior', {}).get(f'uses_{behavior_key}', 0))
                    
                elif feat_name.startswith('cfg_'):
                    cfg_key = feat_name.replace('cfg_', '')
                    features.append(ir.get('cfg', {}).get(cfg_key, 0))
                    
                elif feat_name == 'num_unique_apis':
                    features.append(len(ir.get('apis', [])))
                    
                elif feat_name == 'total_api_calls':
                    features.append(sum(api.get('count', 0) for api in ir.get('apis', [])))
                    
                elif feat_name.startswith('top_api_'):
                    idx = int(feat_name.split('_')[-2]) - 1
                    apis = sorted(ir.get('apis', []), key=lambda x: x.get('count', 0), reverse=True)
                    features.append(apis[idx]['count'] if idx < len(apis) else 0)
                    
                elif feat_name == 'num_unique_opcodes':
                    features.append(len(ir.get('opcodes', [])))
                    
                elif feat_name == 'total_opcodes':
                    features.append(sum(op.get('count', 0) for op in ir.get('opcodes', [])))
                    
                elif feat_name.startswith('opcode_'):
                    opcode_name = feat_name.replace('opcode_', '').replace('_count', '').upper()
                    opcodes = {op['name']: op['count'] for op in ir.get('opcodes', [])}
                    features.append(opcodes.get(opcode_name, 0))
                    
                elif feat_name in ['call_ratio', 'jmp_ratio', 'api_to_opcode_ratio']:
                    total_opcodes = sum(op.get('count', 0) for op in ir.get('opcodes', []))
                    if total_opcodes > 0:
                        opcodes = {op['name']: op['count'] for op in ir.get('opcodes', [])}
                        if feat_name == 'call_ratio':
                            features.append(opcodes.get('CALL', 0) / total_opcodes)
                        elif feat_name == 'jmp_ratio':
                            features.append(opcodes.get('JMP', 0) / total_opcodes)
                        elif feat_name == 'api_to_opcode_ratio':
                            total_apis = sum(api.get('count', 0) for api in ir.get('apis', []))
                            features.append(total_apis / total_opcodes)
                    else:
                        features.append(0.0)
                else:
                    features.append(0)
            
            feature_array = np.array(features).reshape(1, -1)
            
            # Diagnostic output
            n_nonzero = np.count_nonzero(feature_array)
            
            # Hash the IR file to verify it's unique
            import hashlib
            ir_hash = hashlib.md5(str(ir).encode()).hexdigest()[:8]
            
            print(f"[*] IR File Hash: {ir_hash}")
            print(f"[*] Extracted {len(features)} features")
            print(f"    Feature range: [{feature_array.min():.2f}, {feature_array.max():.2f}]")
            print(f"    Non-zero features: {n_nonzero}")
            print(f"    First 5 features: {feature_array[0][:5]}")
            print(f"    Feature hash: {hashlib.md5(feature_array.tobytes()).hexdigest()[:8]}")
            
            if n_nonzero == 0:
                print("[!] WARNING: All features are zero! Parsing likely failed or file is empty.")
            
            return feature_array
            
        except Exception as e:
            print(f"[✗] Error extracting features: {e}")
            return None
    
    def predict(self, features, temperature=2.5):
        """Make prediction with temperature scaling to soften probabilities"""
        if self.scaler is not None:
            features_input = self.scaler.transform(features)
        else:
            features_input = features

        prediction = self.model.predict(features_input)[0]
        raw_probability = self.model.predict_proba(features_input)[0]
        
        # Apply temperature scaling to soften extreme probabilities
        # Higher temperature = softer probabilities (more uncertainty)
        # T=1.0 means no scaling, T>1 softens, T<1 sharpens
        import numpy as np
        logits = np.log(raw_probability + 1e-10)  # Add small epsilon to avoid log(0)
        scaled_logits = logits / temperature
        
        # Convert back to probabilities using softmax
        exp_logits = np.exp(scaled_logits)
        probability = exp_logits / np.sum(exp_logits)
        
        # Explicit logging to debug calibration
        print(f"\n[PREDICTION DEBUG]")
        print(f"  Raw probabilities: Benign={raw_probability[0]:.6f}, Malware={raw_probability[1]:.6f}")
        print(f"  Temperature scaled (T={temperature}): Benign={probability[0]:.6f}, Malware={probability[1]:.6f}")
        print(f"  Prediction: {'MALWARE' if prediction == 1 else 'BENIGN'}")
        print(f"  Model type: {type(self.model).__name__}")
        
        return prediction, probability
    
    def explain_prediction(self, features, top_n=5):
        """Explain why a sample was classified as it was"""
        try:
            feature_names = self.metadata.get('feature_names', self.metadata.get('features', []))
            
            # Handle both RF and Calibrated RF models
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                # LDA uses coef_
                importances = np.abs(self.model.coef_[0])
            else:
                return []
            
            if self.scaler:
                features_scaled = self.scaler.transform(features)[0]
            else:
                features_scaled = features[0]
            
            contributions = []
            descriptions = {
                'uses_network': "Network activity detected",
                'uses_fileops': "File system operations detected",
                'uses_registry': "Registry manipulation detected",
                'uses_memory': "Direct memory manipulation",
                'uses_injection': "Code injection capabilities",
                'uses_crypto': "Cryptographic operations detected",
                'uses_persist': "Persistence mechanisms detected",
                'cfg_num_blocks': "Complex control flow",
                'cfg_cyclomatic_complexity': "High code complexity",
                'num_unique_apis': "High number of unique APIs",
                'total_api_calls': "High volume of API calls",
            }
            
            for i, (name, importance) in enumerate(zip(feature_names, importances)):
                val = features[0][i]
                scaled_val = features_scaled[i]
                score = importance * max(0, scaled_val)
                
                base_desc = descriptions.get(name, name.replace('_', ' ').title())
                if name.startswith('uses_'):
                    desc = base_desc if val > 0 else f"No {base_desc.lower()}"
                elif 'count' in name or 'num' in name:
                    desc = f"{base_desc} ({int(val)})"
                else:
                    desc = f"{base_desc} ({val:.2f})"
                
                contributions.append({
                    'name': name,
                    'importance': importance,
                    'value': val,
                    'score': score,
                    'description': desc
                })
            
            contributions.sort(key=lambda x: x['score'], reverse=True)
            return contributions[:top_n]
            
        except Exception as e:
            print(f"[!] Explanation failed: {e}")
            return []
    
    def display_results(self, prediction, probability, input_path):
        """Display prediction results"""
        print("\n╔══════════════════════════════════════════════════════════╗")
        print("║                   Prediction Results                     ║")
        print("╠══════════════════════════════════════════════════════════╣")
        print(f"║ File: {Path(input_path).name[:50]:<50} ║")
        print("╠══════════════════════════════════════════════════════════╣")
        
        if prediction == 1:
            print(f"║ Classification: 🚨 MALICIOUS                             ║")
            print(f"║ Confidence:     {probability[1]*100:5.2f}%                                  ║")
        else:
            print(f"║ Classification: ✅ BENIGN                                ║")
            print(f"║ Confidence:     {probability[0]*100:5.2f}%                                  ║")
        
        print("╠══════════════════════════════════════════════════════════╣")
        print(f"║ Malicious probability: {probability[1]*100:5.2f}%                          ║")
        print(f"║ Benign probability:    {probability[0]*100:5.2f}%                          ║")
        print("╚══════════════════════════════════════════════════════════╝")
        
        if probability[1] >= 0.9:
            risk = "🔴 CRITICAL"
        elif probability[1] >= 0.7:
            risk = "🟠 HIGH"
        elif probability[1] >= 0.5:
            risk = "🟡 MEDIUM"
        else:
            risk = "🟢 LOW"
        
        print(f"\nRisk Level: {risk}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <file.exe|file.asm>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if not Path(input_path).exists():
        print(f"[✗] File not found: {input_path}")
        sys.exit(1)
    
    predictor = MalwarePredictor()
    
    if not predictor.load_model():
        sys.exit(1)
    
    ir_file = predictor.process_sample(input_path)
    if ir_file is None:
        sys.exit(1)
    
    features = predictor.extract_features(ir_file)
    if features is None:
        sys.exit(1)
    
    prediction, probability = predictor.predict(features)
    predictor.display_results(prediction, probability, input_path)
