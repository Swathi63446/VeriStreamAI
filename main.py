"""
Main entry point - runs training script
"""

import os
import subprocess

if __name__ == "__main__":
    # Run the training script
    result = subprocess.run(['python', 'train.py'], cwd=os.path.dirname(__file__))
    exit(result.returncode)