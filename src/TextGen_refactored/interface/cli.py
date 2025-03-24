#!/usr/bin/env python3
"""
Command line interface for the DramaTron Streamlit app.

This module provides a command line interface for launching the DramaTron Streamlit app.
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path
import dotenv


def find_streamlit() -> str:
    """Find the Streamlit executable.
    
    Returns:
        Path to the Streamlit executable
    
    Raises:
        FileNotFoundError: If Streamlit executable is not found
    """
    # Try to find streamlit in common locations
    streamlit_path = subprocess.run(
        ["which", "streamlit"], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True
    ).stdout.strip()
    
    if streamlit_path:
        return streamlit_path
    
    # Check in the virtual environment
    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path:
        streamlit_exe = Path(venv_path) / "bin" / "streamlit"
        if streamlit_exe.exists():
            return str(streamlit_exe)
    
    # Try to find in the Python environment
    python_path = sys.executable
    streamlit_exe = Path(python_path).parent / "streamlit"
    if streamlit_exe.exists():
        return str(streamlit_exe)
    
    # Raise an error if not found
    raise FileNotFoundError(
        "Could not find Streamlit executable. Please ensure it is installed "
        "and available in your PATH environment variable."
    )


def run_app(port: int = 8501, browser: bool = True, env_file: str = None) -> None:
    """Run the Streamlit app.
    
    Args:
        port: Port to run the app on
        browser: Whether to open a browser
        env_file: Path to a .env file with API keys
    """
    # Load environment variables from .env file
    if env_file:
        env_path = Path(env_file)
        if env_path.exists():
            dotenv.load_dotenv(env_path)
            print(f"Loaded environment variables from {env_path}")
        else:
            print(f"Warning: Environment file {env_path} not found.")
    
    # Find the Streamlit app
    app_dir = Path(__file__).parent
    app_path = app_dir / "streamlit_app.py"
    
    if not app_path.exists():
        raise FileNotFoundError(f"Streamlit app not found at {app_path}")
    
    # Find the Streamlit executable
    try:
        streamlit = find_streamlit()
    except FileNotFoundError:
        print("Streamlit not found. Trying to launch using module.")
        cmd = [
            sys.executable, 
            "-m", 
            "streamlit", 
            "run", 
            str(app_path), 
            "--server.port", 
            str(port)
        ]
        
        if not browser:
            cmd.extend(["--server.headless", "true"])
    else:
        cmd = [
            streamlit, 
            "run", 
            str(app_path), 
            "--server.port", 
            str(port)
        ]
        
        if not browser:
            cmd.extend(["--server.headless", "true"])
    
    # Run the app
    print(f"Launching DramaTron at http://localhost:{port}")
    subprocess.run(cmd)


def main() -> None:
    """Main function for the CLI."""
    parser = argparse.ArgumentParser(
        description="Launch the DramaTron Streamlit web app."
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8501, 
        help="Port to run the app on"
    )
    parser.add_argument(
        "--no-browser", 
        action="store_true", 
        help="Do not open a browser"
    )
    parser.add_argument(
        "--env-file", 
        type=str, 
        help="Path to a .env file with API keys"
    )
    
    args = parser.parse_args()
    
    try:
        run_app(port=args.port, browser=not args.no_browser, env_file=args.env_file)
    except Exception as e:
        print(f"Error launching DramaTron: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 