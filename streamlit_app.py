#!/usr/bin/env python3
"""
Streamlit Interface for MEEF Malware Detection
Provides a web UI for analyzing malware samples
"""

import streamlit as st
import subprocess
import sys
import tempfile
from pathlib import Path
import os

# Import existing modules
sys.path.append(str(Path(__file__).parent))
from validate_asm import is_binary
from data.models.predict import MalwarePredictor


def setup_page():
    """Configure Streamlit page"""
    st.set_page_config(
        page_title="MEEF Malware Detector",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        /* Main header */
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 1rem 0;
        }
        
        /* Subtitle */
        .subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #4a5568;
            margin-bottom: 2rem;
        }
        
        /* Result boxes with dark backgrounds for better contrast */
        .result-box {
            padding: 2rem;
            border-radius: 15px;
            margin: 1.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .malicious {
            background: linear-gradient(135deg, #742a2a 0%, #9b2c2c 100%);
            border-left: 6px solid #fc8181;
            color: #fff;
        }
        .benign {
            background: linear-gradient(135deg, #22543d 0%, #276749 100%);
            border-left: 6px solid #68d391;
            color: #fff;
        }
        
        /* Metric cards */
        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
        }
        div[data-testid="metric-container"] label {
            color: #e0e0e0 !important;
        }
        div[data-testid="metric-container"] div {
            color: white !important;
        }
        
        /* Progress bars */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            font-size: 1.1rem;
            font-weight: bold;
            border-radius: 10px;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
        }
        
        /* Sidebar - Dark Theme */
        section[data-testid="stSidebar"] {
            background-color: #1a202c;
            color: #e2e8f0;
        }
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {
            color: #f7fafc !important;
        }
        section[data-testid="stSidebar"] p, 
        section[data-testid="stSidebar"] li,
        section[data-testid="stSidebar"] div {
            color: #cbd5e0 !important;
        }
        </style>
    """, unsafe_allow_html=True)


def display_header():
    """Display app header"""
    st.markdown('<h1 class="main-header">🛡️ MEEF Malware Detector</h1>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <div class='subtitle'>
            Upload an executable or assembly file to analyze for malware
        </div>
    """, unsafe_allow_html=True)
    st.markdown("")


def validate_asm_file(filepath):
    """Validate if file is a valid ASM file"""
    try:
        if is_binary(filepath):
            return False, "File is binary, needs disassembly"
        
        opcodes = ['MOV', 'CALL', 'JMP', 'PUSH', 'POP', 'RET', 'ADD', 'SUB']
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().upper()
            opcode_found = any(op in content for op in opcodes)
        
        if opcode_found:
            return True, "Valid ASM file detected"
        else:
            return False, "No assembly opcodes found"
            
    except Exception as e:
        return False, f"Error validating file: {e}"


def find_bash():
    """Find bash executable - prioritize WSL on Windows"""
    if os.name == 'nt':
        try:
            result = subprocess.run(['wsl', '--status'], capture_output=True, timeout=3)
            if result.returncode == 0 or b'Default' in result.stdout:
                return 'wsl'
        except:
            pass
        return None
    else:
        return 'bash'


def check_objdump_available(bash_path):
    """Check if objdump is available"""
    try:
        if bash_path == 'wsl':
            result = subprocess.run(['wsl', 'which', 'objdump'], capture_output=True, text=True, timeout=5)
        else:
            result = subprocess.run([bash_path, '-c', 'which objdump'], capture_output=True, text=True, timeout=5)
        return result.returncode == 0 and result.stdout.strip() != ''
    except:
        return False


def windows_path_to_wsl(windows_path):
    """Convert Windows path to WSL path"""
    windows_path = str(Path(windows_path).resolve())
    if len(windows_path) >= 2 and windows_path[1] == ':':
        drive = windows_path[0].lower()
        rest = windows_path[2:].replace('\\', '/')
        return f"/mnt/{drive}{rest}"
    return windows_path.replace('\\', '/')


def disassemble_exe(exe_path, output_path):
    """Disassemble an executable using objdump via bash/WSL"""
    try:
        bash_path = find_bash()
        if not bash_path:
            return False, "WSL not found. Please install WSL to analyze .exe files."
        
        script_dir = Path(__file__).parent
        disasm_script = script_dir / 'disassemble.sh'
        
        if not disasm_script.exists():
            return False, f"Disassembly script not found: {disasm_script}"
        
        # Build command with line ending fix
        if bash_path == 'wsl':
            wsl_script_path = windows_path_to_wsl(disasm_script)
            wsl_exe_path = windows_path_to_wsl(exe_path)
            wsl_output_path = windows_path_to_wsl(output_path)
            # Fix line endings with sed before running
            cmd_str = f"sed 's/\\r$//' '{wsl_script_path}' | bash -s '{wsl_exe_path}' '{wsl_output_path}'"
            cmd = ['wsl', 'bash', '-c', cmd_str]
        else:
            cmd = [bash_path, str(disasm_script), str(exe_path), str(output_path)]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=60
        )
        
        if Path(output_path).exists() and Path(output_path).stat().st_size > 100:
            return True, "Disassembly successful" + (" (via WSL)" if bash_path == 'wsl' else "")
        
        error_msg = result.stderr or result.stdout or "Unknown error"
        return False, f"Disassembly failed: {error_msg[:500]}"
        
    except subprocess.TimeoutExpired:
        return False, "Disassembly timed out (>60s)"
    except Exception as e:
        return False, f"Error during disassembly: {e}"


def run_prediction(asm_path):
    """Run malware prediction using predict.py"""
    try:
        if not Path(asm_path).exists():
            return None, f"ASM file not found: {asm_path}"
        
        predictor = MalwarePredictor()
        
        if not predictor.load_model():
            return None, "Failed to load ML model"
        
        ir_file = predictor.process_sample(asm_path)
        if ir_file is None:
            return None, "Failed to process sample"
        
        features = predictor.extract_features(ir_file)
        if features is None:
            return None, "Failed to extract features"
        
        prediction, probability = predictor.predict(features)
        explanation = predictor.explain_prediction(features)
        
        result = {
            'prediction': int(prediction),
            'is_malware': bool(prediction == 1),
            'malware_prob': float(probability[1]),
            'benign_prob': float(probability[0]),
            'confidence': float(max(probability)),
            'explanation': explanation
        }
        
        return result, "Prediction successful"
        
    except Exception as e:
        return None, f"Error during prediction: {e}"


def display_results(result):
    """Display prediction results"""
    is_malware = result['is_malware']
    malware_prob = result['malware_prob']
    benign_prob = result['benign_prob']
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if is_malware:
            st.markdown(f"""
                <div class='result-box malicious'>
                    <h2 style='color: #fff; text-align: center;'>🚨 MALICIOUS DETECTED</h2>
                    <p style='text-align: center; font-size: 1.5rem; color: #fff;'>
                        Confidence: <strong>{malware_prob*100:.2f}%</strong>
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='result-box benign'>
                    <h2 style='color: #fff; text-align: center;'>✅ BENIGN DETECTED</h2>
                    <p style='text-align: center; font-size: 1.5rem; color: #fff;'>
                        Confidence: <strong>{benign_prob*100:.2f}%</strong>
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    # Detailed probabilities
    st.markdown("### Detailed Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Malware Probability", f"{malware_prob*100:.2f}%")
    with col2:
        st.metric("Benign Probability", f"{benign_prob*100:.2f}%")
    
    st.progress(malware_prob, text=f"Malware: {malware_prob*100:.1f}%")
    st.progress(benign_prob, text=f"Benign: {benign_prob*100:.1f}%")

    # Risk level
    if malware_prob >= 0.9:
        risk = "🔴 CRITICAL"
        risk_color = "#ff0000"
    elif malware_prob >= 0.7:
        risk = "🟠 HIGH"
        risk_color = "#ff6600"
    elif malware_prob >= 0.5:
        risk = "🟡 MEDIUM"
        risk_color = "#ffaa00"
    else:
        risk = "🟢 LOW"
        risk_color = "#00cc00"
    
    st.markdown(f"""
        <div style='text-align: center; padding: 1rem; font-size: 1.5rem; margin-top: 1rem;'>
            <strong>Risk Level:</strong> <span style='color: {risk_color}; font-weight: bold;'>{risk}</span>
        </div>
    """, unsafe_allow_html=True)

    # Threat Indicators
    if 'explanation' in result and result['explanation']:
        st.markdown("---")
        st.subheader("🔍 Threat Indicators")
        
        if is_malware:
            st.warning("The following suspicious patterns were detected:")
        else:
            st.info("Key features observed:")
            
        for item in result['explanation']:
            desc = item['description']
            relevance = min(1.0, item['score'] * 10)
            
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"**{desc}**")
                st.caption(f"Technical Feature: `{item['name']}`")
            with col_b:
                st.progress(relevance)


def main():
    """Main Streamlit application"""
    setup_page()
    display_header()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
            This tool analyzes executable files and assembly code to detect malware using machine learning.
            
            **Workflow:**
            1. Upload .exe or .asm file
            2. Validate/Convert to ASM
            3. Extract features
            4. ML prediction
            
            **Supported formats:**
            - `.exe` (Windows executables)
            - `.asm` (Assembly files)
            - `.dll` (Dynamic libraries)
        """)
        
        st.markdown("---")
        st.markdown("**Powered by MEEF**")
        st.markdown("Machine Learning Enhanced Executable Forensics")
        
        # Diagnostic information
        st.markdown("---")
        st.markdown("**🔧 System Diagnostics**")
        bash_path = find_bash()
        
        if os.name == 'nt':
            if bash_path == 'wsl':
                st.success("✅ WSL detected")
                if check_objdump_available(bash_path):
                    st.success("✅ objdump available in WSL")
                else:
                    st.error("❌ objdump not found in WSL")
                    st.caption("Run in WSL: sudo apt install binutils")
            else:
                st.error("❌ WSL not found")
                st.caption("Install WSL to analyze .exe files")
        else:
            if bash_path:
                st.success("✅ Bash found")
                if check_objdump_available(bash_path):
                    st.success("✅ objdump available")
                else:
                    st.error("❌ objdump not found")
    
    # Main content
    st.header("📁 Upload File")
    
    uploaded_file = st.file_uploader(
        "Choose a file to analyze",
        type=['exe', 'asm', 'dll'],
        help="Upload an executable (.exe, .dll) or assembly (.asm) file"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: **{uploaded_file.name}** ({uploaded_file.size} bytes)")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            uploaded_path = temp_path / uploaded_file.name
            with open(uploaded_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("🔍 Analyze File", type="primary"):
                with st.spinner("Analyzing file..."):
                    
                    progress_bar = st.progress(0, text="Starting analysis...")
                    status_container = st.empty()
                    
                    # Step 1: Validate/Convert
                    status_container.info("**Step 1/3:** Validating file format...")
                    progress_bar.progress(10)
                    
                    file_ext = uploaded_path.suffix.lower()
                    asm_path = uploaded_path
                    
                    if file_ext == '.asm':
                        is_valid, msg = validate_asm_file(uploaded_path)
                        if is_valid:
                            status_container.success(f"✅ {msg}")
                        else:
                            status_container.error(f"❌ {msg}")
                            st.stop()
                    
                    elif file_ext in ['.exe', '.dll']:
                        status_container.info("**Step 1/3:** Converting to assembly...")
                        progress_bar.progress(20)
                        
                        asm_path = temp_path / (uploaded_path.stem + '.asm')
                        success, msg = disassemble_exe(uploaded_path, asm_path)
                        
                        if success:
                            status_container.success(f"✅ {msg}")
                        else:
                            status_container.error(f"❌ Disassembly Failed")
                            st.error(msg)
                            st.stop()
                    
                    else:
                        status_container.error(f"❌ Unsupported file type: {file_ext}")
                        st.stop()
                    
                    progress_bar.progress(40)
                    
                    # Step 2: Show ASM preview
                    status_container.info("**Step 2/3:** Analyzing assembly code...")
                    progress_bar.progress(50)
                    
                    with st.expander("📄 View Assembly Code Preview"):
                        try:
                            with open(asm_path, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = f.read().split('\n')
                                st.code('\n'.join(lines[:50]), language='asm')
                                if len(lines) > 50:
                                    st.info(f"Showing first 50 lines of {len(lines)} total")
                        except Exception as e:
                            st.warning(f"Could not preview: {e}")
                    
                    progress_bar.progress(60)
                    
                    # Step 3: Run prediction
                    status_container.info("**Step 3/3:** Running malware detection...")
                    progress_bar.progress(70)
                    
                    result, msg = run_prediction(asm_path)
                    
                    if result is None:
                        status_container.error(f"❌ {msg}")
                        st.stop()
                    
                    progress_bar.progress(100, text="Analysis complete! ✅")
                    status_container.success("✅ Analysis complete!")
                    
                    # Display results
                    st.markdown("---")
                    st.header("📊 Results")
                    display_results(result)
                    
                    # Download option
                    if file_ext in ['.exe', '.dll']:
                        st.markdown("---")
                        st.subheader("💾 Download Disassembled File")
                        with open(asm_path, 'r', encoding='utf-8', errors='ignore') as f:
                            st.download_button(
                                label="Download .asm file",
                                data=f.read(),
                                file_name=asm_path.name,
                                mime="text/plain"
                            )
    
    else:
        st.info("""
            👆 **Upload a file above to get started**
            
            The analyzer will:
            1. Disassemble executables (if needed)
            2. Parse assembly code into IR
            3. Extract behavioral features
            4. Run ML prediction
            5. Show threat indicators
        """)


if __name__ == "__main__":
    main()
