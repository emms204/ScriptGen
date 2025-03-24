#!/usr/bin/env python3
"""
Script to run the DramaTron UI.

This is a simple script to launch the DramaTron UI using Streamlit.
"""
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import the CLI module
from TextGen_refactored.interface.cli import main


if __name__ == "__main__":
    # Check if .env file exists and load it
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        env_arg = f"--env-file={env_file}"
        sys.argv.append(env_arg)
    
    # Run the CLI
    main() 