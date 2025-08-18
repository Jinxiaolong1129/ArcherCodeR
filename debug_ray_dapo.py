#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Debug script for Ray DAPO training
This script modifies the existing training script to enable Ray debugging
"""

import os
import sys
import subprocess
import tempfile

def create_debug_training_script():
    """Create a modified version of the training script with Ray debugging enabled"""
    
    # Read the original script
    original_script_path = "scripts/train/run_archer_qwen2.5_1.5b_code_single.sh"
    
    with open(original_script_path, 'r') as f:
        script_content = f.read()
    
    # Add Ray debugging environment variables after the existing exports
    debug_exports = """
# Ray Debugging Environment Variables
export RAY_DEBUG_POST_MORTEM=1

"""
    
    # Find the position after the existing exports and add our debug exports
    lines = script_content.split('\n')
    new_lines = []
    exports_added = False
    
    for line in lines:
        new_lines.append(line)
        # Add debug exports after the WANDB_API_KEY export
        if line.startswith('export WANDB_API_KEY=') and not exports_added:
            new_lines.append('')
            new_lines.append('# Ray Debugging Environment Variables')
            new_lines.append('export RAY_DEBUG_POST_MORTEM=1')
            new_lines.append('')
            exports_added = True
        
        # Modify the experiment name to indicate debugging
        if line.startswith("exp_name='Archer-Qwen2.5-1.5B-Single'"):
            new_lines[-1] = "exp_name='Archer-Qwen2.5-1.5B-Single-Debug'"
        
        # Reduce total epochs for debugging
        if 'trainer.total_epochs=10' in line:
            new_lines[-1] = line.replace('trainer.total_epochs=10', 'trainer.total_epochs=1')
    
    return '\n'.join(new_lines)

def main():
    print("🚀 Creating Ray DAPO debugging script...")
    print("📍 This will run the training with Ray debugging enabled")
    print()
    
    # Create a temporary debug script
    debug_script_content = create_debug_training_script()
    
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(debug_script_content)
        temp_script_path = f.name
    
    try:
        # Make the script executable
        os.chmod(temp_script_path, 0o755)
        
        print("📝 Ray Debugging Instructions:")
        print("   1. Ray cluster will start and may hit breakpoints in @ray.remote functions")
        print("   2. Open Ray Distributed Debugger extension in VSCode sidebar")
        print("   3. Add your Ray cluster (default: http://127.0.0.1:8265)")
        print("   4. If any breakpoints are hit, click on paused tasks to attach debugger")
        print("   5. Use VSCode debugging features normally")
        print()
        print("🎯 Starting modified training script with debugging...")
        print(f"📁 Temporary debug script: {temp_script_path}")
        print()
        
        # Run the modified script
        result = subprocess.run(['bash', temp_script_path], 
                              cwd='/workspace/ArcherCodeR',
                              env=os.environ.copy())
        
        return result.returncode
        
    finally:
        # Clean up the temporary file
        try:
            os.unlink(temp_script_path)
        except:
            pass

if __name__ == "__main__":
    sys.exit(main()) 