#!/usr/bin/env python3
"""
Environment Setup Script for Audio to Text Converter
This script sets up a virtual environment and installs dependencies
"""

import os
import sys
import subprocess
import venv
from pathlib import Path

def create_virtual_environment():
    """Create a virtual environment in the project directory"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists!")
        return venv_path
    
    print("🔧 Creating virtual environment...")
    venv.create(venv_path, with_pip=True)
    print("✅ Virtual environment created successfully!")
    return venv_path

def get_pip_path(venv_path):
    """Get the pip executable path for the virtual environment"""
    if os.name == 'nt':  # Windows
        return venv_path / "Scripts" / "pip.exe"
    else:  # Unix/Linux/macOS
        return venv_path / "bin" / "pip"

def install_dependencies(venv_path):
    """Install required dependencies"""
    pip_path = get_pip_path(venv_path)
    
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([
            str(pip_path), "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    return True

def main():
    """Main setup function"""
    print("=" * 60)
    print("🚀 AUDIO TO TEXT CONVERTER - ENVIRONMENT SETUP")
    print("=" * 60)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        return
    
    # Create virtual environment
    venv_path = create_virtual_environment()
    
    # Install dependencies
    if install_dependencies(venv_path):
        print("\n" + "=" * 60)
        print("✅ SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📋 Next steps:")
        print("1. Activate the virtual environment:")
        if os.name == 'nt':  # Windows
            print("   venv\\Scripts\\activate")
        else:  # Unix/Linux/macOS
            print("   source venv/bin/activate")
        print("2. Run the audio converter:")
        print("   python audio_to_text.py")
        print("=" * 60)
    else:
        print("❌ Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
