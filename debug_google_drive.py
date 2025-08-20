#!/usr/bin/env python3
"""
Debug script for Google Drive setup and permissions.
Run this to check if your Google Drive integration is working correctly.
"""

import os
import sys

# Add the current directory to the path so we can import google_drive_sync
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from google_drive_sync import GoogleDriveSync
    print("✅ Successfully imported GoogleDriveSync")
except ImportError as e:
    print(f"❌ Failed to import GoogleDriveSync: {e}")
    print("Make sure you have the required packages installed:")
    print("pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
    sys.exit(1)

def main():
    print("=== Google Drive Debug Script ===")
    print()
    
    # Check for credentials file
    if not os.path.exists('credentials.json'):
        print("❌ credentials.json not found!")
        print("Please download your Google Cloud credentials file and save it as 'credentials.json'")
        print("You can get it from: https://console.cloud.google.com/apis/credentials")
        return
    
    print("✅ credentials.json found")
    
    try:
        # Initialize Google Drive sync
        print("\n🔄 Initializing Google Drive sync...")
        drive_sync = GoogleDriveSync()
        print("✅ Google Drive sync initialized successfully")
        
        # Run debug setup
        print("\n🔍 Running debug setup...")
        success = drive_sync.debug_drive_setup()
        
        if success:
            print("\n✅ Debug completed successfully")
            print("\nNext steps:")
            print("1. Check the output above to see if your folder and files are visible")
            print("2. If files are not found, check the folder ID and permissions")
            print("3. Try running the sync again after fixing any issues")
        else:
            print("\n❌ Debug failed - check the error messages above")
            
    except Exception as e:
        print(f"\n❌ Error during debug: {e}")
        print("\nCommon issues:")
        print("1. Invalid credentials.json file")
        print("2. Network connectivity issues")
        print("3. Google Drive API not enabled")
        print("4. Insufficient permissions")

if __name__ == "__main__":
    main() 