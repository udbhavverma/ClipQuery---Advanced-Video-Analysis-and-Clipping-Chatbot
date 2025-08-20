#!/usr/bin/env python3
"""
Helper script to install FFmpeg on Windows
"""

import subprocess
import sys
import os

def check_chocolatey():
    """Check if Chocolatey is installed"""
    try:
        result = subprocess.run(["choco", "--version"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_chocolatey():
    """Install Chocolatey package manager"""
    print("📦 Installing Chocolatey...")
    try:
        # Run PowerShell as administrator to install Chocolatey
        cmd = [
            "powershell", "-Command", 
            "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Chocolatey installed successfully!")
            return True
        else:
            print(f"❌ Failed to install Chocolatey: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing Chocolatey: {e}")
        return False

def install_ffmpeg():
    """Install FFmpeg using Chocolatey"""
    print("🎬 Installing FFmpeg...")
    try:
        result = subprocess.run(["choco", "install", "ffmpeg", "-y"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg installed successfully!")
            return True
        else:
            print(f"❌ Failed to install FFmpeg: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing FFmpeg: {e}")
        return False

def check_ffmpeg():
    """Check if FFmpeg is now available"""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    print("🎬 FFmpeg Installation Helper")
    print("=" * 40)
    
    # Check if FFmpeg is already installed
    if check_ffmpeg():
        print("✅ FFmpeg is already installed!")
        return
    
    print("❌ FFmpeg not found. Installing...")
    
    # Check if Chocolatey is installed
    if not check_chocolatey():
        print("📦 Chocolatey not found. Installing Chocolatey first...")
        if not install_chocolatey():
            print("\n❌ Failed to install Chocolatey.")
            print("Please install FFmpeg manually:")
            print("1. Download from: https://ffmpeg.org/download.html")
            print("2. Extract to a folder (e.g., C:\\ffmpeg)")
            print("3. Add C:\\ffmpeg\\bin to your system PATH")
            return
    
    # Install FFmpeg
    if install_ffmpeg():
        print("\n🔄 Refreshing environment variables...")
        # Refresh PATH by restarting the shell
        print("✅ FFmpeg installation completed!")
        print("\n📝 Next steps:")
        print("1. Close this terminal/command prompt")
        print("2. Open a new terminal/command prompt")
        print("3. Run: python run_chatbot.py")
        print("4. FFmpeg should now be available for video transcription")
    else:
        print("\n❌ Failed to install FFmpeg automatically.")
        print("Please install FFmpeg manually:")
        print("1. Download from: https://ffmpeg.org/download.html")
        print("2. Extract to a folder (e.g., C:\\ffmpeg)")
        print("3. Add C:\\ffmpeg\\bin to your system PATH")

if __name__ == "__main__":
    main() 