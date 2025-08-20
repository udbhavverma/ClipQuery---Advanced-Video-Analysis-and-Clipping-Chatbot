import os
import json
import time
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
import tempfile
import shutil

# Google Drive API imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.errors import HttpError

# For file watching and transcription
import whisper
from pathlib import Path

# Import the enhanced VectorStore from rag_pipeline
from rag_pipeline import VectorStore

import re
import pickle

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly', 'https://www.googleapis.com/auth/drive.file']

class GoogleDriveSync:
    def __init__(self, credentials_file: str = "credentials.json", token_file: str = "token.json"):
        """
        Initialize Google Drive sync.
        
        Args:
            credentials_file: Path to Google Cloud credentials JSON file
            token_file: Path to store authentication token
        """
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.local_sync_dir = "Max Life Videos"
        self.sync_interval = 30  # seconds
        self.is_watching = False
        self.watch_thread = None
        self.folder_id = None  # Will be set by find_or_create_folder()
        
        # Initialize Whisper model for transcription
        try:
            self.whisper_model = whisper.load_model("base")
            print("✅ Whisper base model loaded successfully")
            
            # Also try to load a smaller model for faster transcription
            try:
                self.whisper_model_small = whisper.load_model("tiny")
                print("✅ Whisper tiny model loaded for faster transcription")
            except Exception as e:
                print(f"⚠️ Could not load Whisper tiny model: {e}")
                self.whisper_model_small = None
                
        except Exception as e:
            print(f"⚠️ Warning: Could not load Whisper model: {e}")
            self.whisper_model = None
            self.whisper_model_small = None
        
        # Authenticate with Google Drive
        try:
            self.service = self.authenticate()
            print("✅ Google Drive authentication successful")
        except Exception as e:
            print(f"❌ Google Drive authentication failed: {e}")
            raise
        
        # Create local sync directory if it doesn't exist
        os.makedirs(self.local_sync_dir, exist_ok=True)
        
    def authenticate(self):
        """Authenticate with Google Drive API."""
        creds = None
        
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.json'):
            try:
                creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            except Exception as e:
                print(f"⚠️ Error loading existing token: {e}")
                print("🗑️ Removing invalid token file...")
                os.remove('token.json')
                creds = None
        
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"⚠️ Error refreshing token: {e}")
                    print("🗑️ Removing invalid token file...")
                    if os.path.exists('token.json'):
                        os.remove('token.json')
                    creds = None
            
            if not creds:
                print("🔐 Starting Google Drive authentication...")
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'credentials.json', SCOPES)
                    creds = flow.run_local_server(port=0)
                    # Save the credentials for the next run
                    with open('token.json', 'w') as token:
                        token.write(creds.to_json())
                    print("✅ Authentication successful!")
                except Exception as e:
                    print(f"❌ Authentication failed: {e}")
                    raise
        
        return build('drive', 'v3', credentials=creds)
    
    def find_or_create_folder(self, folder_name: str = "Max Life Videos") -> str:
        """
        Find or create a folder in Google Drive.
        
        Args:
            folder_name: Name of the folder to find/create
            
        Returns:
            Folder ID
        """
        # Search for existing folder
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = self.service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        files = results.get('files', [])
        
        if files:
            self.folder_id = files[0]['id']
            print(f"✅ Found existing folder: {folder_name} (ID: {self.folder_id})")
            return self.folder_id
        
        # Create new folder if not found
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = self.service.files().create(body=folder_metadata, fields='id').execute()
        self.folder_id = folder.get('id')
        print(f"✅ Created new folder: {folder_name} (ID: {self.folder_id})")
        return self.folder_id
    
    def list_drive_files(self) -> List[Dict[str, Any]]:
        """
        List all files in the Google Drive folder.
        
        Returns:
            List of file metadata dictionaries
        """
        if not self.folder_id:
            self.find_or_create_folder()
        
        query = f"'{self.folder_id}' in parents and trashed=false"
        all_files = []
        page_token = None
        
        while True:
            try:
                results = self.service.files().list(
                    q=query, 
                    spaces='drive', 
                    fields='nextPageToken, files(id, name, mimeType, modifiedTime, size)',
                    orderBy='modifiedTime desc',
                    pageToken=page_token,
                    pageSize=1000  # Maximum page size
                ).execute()
                files = results.get('files', [])
                all_files.extend(files)
                # Check if there are more pages
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
            except Exception as e:
                print(f"❌ Error listing files: {e}")
                break
        
        print(f"📄 Retrieved {len(all_files)} files from Google Drive")
        return all_files
    
    def download_file(self, file_id: str, local_path: str) -> bool:
        """
        Download a file from Google Drive.
        
        Args:
            file_id: Google Drive file ID
            local_path: Local path to save the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            request = self.service.files().get_media(fileId=file_id)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            with open(local_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    if status:
                        print(f"Download {int(status.progress() * 100)}%")
            
            # Verify file integrity
            if not self._verify_file_integrity(local_path):
                print(f"⚠️ File integrity check failed for {os.path.basename(local_path)}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error downloading file: {e}")
            return False

    def _verify_file_integrity(self, file_path: str) -> bool:
        """
        Verify file integrity by checking if file is readable and has content.
        
        Args:
            file_path: Path to the file to verify
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            # Check if file exists and has size > 0
            if not os.path.exists(file_path):
                return False
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                print(f"⚠️ File is empty: {os.path.basename(file_path)}")
                return False
            
            # For video files, try to open with OpenCV to verify integrity
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in video_extensions:
                try:
                    import cv2
                    cap = cv2.VideoCapture(file_path)
                    if not cap.isOpened():
                        print(f"⚠️ Cannot open video file: {os.path.basename(file_path)}")
                        return False
                    cap.release()
                except ImportError:
                    # OpenCV not available, skip video verification
                    pass
                except Exception as e:
                    print(f"⚠️ Video file integrity check failed: {os.path.basename(file_path)} - {e}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ File integrity check error: {e}")
            return False
    
    def upload_file(self, local_path: str, drive_name: str = None) -> Optional[str]:
        """
        Upload a file to Google Drive.
        
        Args:
            local_path: Local path of the file to upload
            drive_name: Name to use in Google Drive (defaults to local filename)
            
        Returns:
            Google Drive file ID if successful, None otherwise
        """
        if not self.folder_id:
            self.find_or_create_folder()
        
        if not drive_name:
            drive_name = os.path.basename(local_path)
        
        try:
            file_metadata = {
                'name': drive_name,
                'parents': [self.folder_id]
            }
            
            media = MediaFileUpload(local_path, resumable=True)
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            print(f"✅ Uploaded {drive_name} (ID: {file.get('id')})")
            return file.get('id')
        except HttpError as error:
            print(f"❌ Error uploading file {local_path}: {error}")
            return None
    
    def sync_from_drive(self) -> Dict[str, List[str]]:
        """
        Sync files from Google Drive to local directory.
        
        Returns:
            Dictionary with 'downloaded', 'updated', and 'errors' lists
        """
        drive_files = self.list_drive_files()
        local_files = set(os.listdir(self.local_sync_dir)) if os.path.exists(self.local_sync_dir) else set()
        
        downloaded = []
        updated = []
        errors = []
        
        # Create a mapping of base names to check for existing transcripts
        base_name_map = {}
        for local_file in local_files:
            base_name = os.path.splitext(local_file)[0]
            if base_name not in base_name_map:
                base_name_map[base_name] = []
            base_name_map[base_name].append(local_file)
        
        for file in drive_files:
            file_name = file['name']
            local_path = os.path.join(self.local_sync_dir, file_name)
            
            # Check if file needs to be downloaded or updated
            should_download = False
            if file_name not in local_files:
                should_download = True
                downloaded.append(file_name)
            else:
                # Check if remote file is newer
                try:
                    local_mtime = os.path.getmtime(local_path) if os.path.exists(local_path) else 0
                    drive_mtime = datetime.fromisoformat(file['modifiedTime'].replace('Z', '+00:00')).timestamp()
                    
                    if drive_mtime > local_mtime:
                        should_download = True
                        updated.append(file_name)
                except Exception as e:
                    print(f"⚠️ Error checking file timestamps for {file_name}: {e}")
                    # If we can't check timestamps, download to be safe
                    should_download = True
                    updated.append(file_name)
            
            if should_download:
                try:
                    # Create a temporary file first to avoid corruption
                    temp_path = local_path + '.tmp'
                    if self.download_file(file['id'], temp_path):
                        # Move temp file to final location
                        if os.path.exists(local_path):
                            os.remove(local_path)
                        os.rename(temp_path, local_path)
                        print(f"✅ Synced: {file_name}")
                    else:
                        errors.append(file_name)
                        # Clean up temp file if it exists
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                except Exception as e:
                    print(f"❌ Error syncing {file_name}: {e}")
                    errors.append(file_name)
                    # Clean up temp file if it exists
                    temp_path = local_path + '.tmp'
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        # After regular sync, check for missing video files
        print("🔍 Checking for missing video files...")
        missing_videos_result = self.check_and_download_missing_videos()
        
        # Add downloaded missing videos to the main result
        downloaded.extend(missing_videos_result['downloaded'])
        errors.extend(missing_videos_result['failed'])
        
        return {
            'downloaded': downloaded,
            'updated': updated,
            'errors': errors,
            'missing_videos_downloaded': missing_videos_result['downloaded'],
            'missing_videos_failed': missing_videos_result['failed'],
            'missing_videos_not_found': missing_videos_result['not_found']
        }

    def check_and_download_missing_videos(self) -> Dict[str, List[str]]:
        """
        Check for missing video files that have transcripts but no MP4 files,
        and attempt to download them from Google Drive.
        
        Returns:
            Dictionary with 'downloaded', 'failed', and 'not_found' lists
        """
        downloaded = []
        failed = []
        not_found = []
        
        if not os.path.exists(self.local_sync_dir):
            return {'downloaded': downloaded, 'failed': failed, 'not_found': not_found}
        
        # Get all transcript files
        transcript_files = []
        for file_name in os.listdir(self.local_sync_dir):
            if file_name.endswith('.txt'):
                transcript_files.append(file_name)
        
        print(f"🔍 Checking {len(transcript_files)} transcript files for missing videos...")
        
        # Debug: Show what folder we're looking in
        print(f"🔍 Looking in Google Drive folder ID: {self.folder_id}")
        
        # Get all files from Google Drive for debugging
        drive_files = self.list_drive_files()
        print(f"🔍 Found {len(drive_files)} files in Google Drive folder")
        
        # Show first few files for debugging
        print("🔍 First 10 files in Google Drive:")
        for i, file_info in enumerate(drive_files[:10]):
            print(f"  {i+1}. {file_info['name']} (ID: {file_info['id']})")
        
        for transcript_file in transcript_files:
            video_id = transcript_file[:-4]  # Remove .txt extension
            video_file_name = f"{video_id}.mp4"
            video_path = os.path.join(self.local_sync_dir, video_file_name)
            
            # Check if video file is missing
            if not os.path.exists(video_path):
                print(f"📥 Video file missing for {video_id}, attempting to download...")
                
                # Look for the video file in Google Drive
                video_file_info = None
                
                for file_info in drive_files:
                    if file_info['name'] == video_file_name:
                        video_file_info = file_info
                        break
                
                if video_file_info:
                    print(f"🎬 Found video in Google Drive: {video_file_name}")
                    
                    # Download the video file
                    if self.download_file(video_file_info['id'], video_path):
                        print(f"✅ Successfully downloaded: {video_file_name}")
                        downloaded.append(video_file_name)
                    else:
                        print(f"❌ Failed to download: {video_file_name}")
                        failed.append(video_file_name)
                else:
                    print(f"⚠️ Video file {video_file_name} not found in Google Drive")
                    # Debug: Show similar files that might exist
                    similar_files = [f['name'] for f in drive_files if video_id.lower() in f['name'].lower()]
                    if similar_files:
                        print(f"   🔍 Similar files found: {similar_files}")
                    not_found.append(video_file_name)
        
        if downloaded:
            print(f"✅ Downloaded {len(downloaded)} missing video files")
        if failed:
            print(f"❌ Failed to download {len(failed)} video files")
        if not_found:
            print(f"⚠️ {len(not_found)} video files not found in Google Drive")
        
        return {
            'downloaded': downloaded,
            'failed': failed,
            'not_found': not_found
        }

    def download_missing_videos_only(self) -> Dict[str, List[str]]:
        """
        Only download missing video files without doing a full sync.
        This is useful when you just want to get missing videos.
        
        Returns:
            Dictionary with download results
        """
        print("🎬 Downloading missing video files only...")
        return self.check_and_download_missing_videos()

    def debug_drive_setup(self):
        """
        Debug Google Drive setup and permissions.
        This will help identify issues with folder access and file visibility.
        """
        print("🔍 Debugging Google Drive setup...")
        
        try:
            # Check authentication
            print("✅ Authentication successful")
            
            # Check folder ID
            if not self.folder_id:
                print("⚠️ No folder ID set, attempting to find/create folder...")
                self.find_or_create_folder()
            
            print(f"📁 Using folder ID: {self.folder_id}")
            
            # List all folders to see what's available
            print("\n📁 Available folders in Google Drive:")
            query = "mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = self.service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
            folders = results.get('files', [])
            
            for i, folder in enumerate(folders):
                print(f"  {i+1}. {folder['name']} (ID: {folder['id']})")
                if folder['id'] == self.folder_id:
                    print(f"      ← This is the current folder")
            
            # List files in the current folder
            print(f"\n📄 Files in current folder (ID: {self.folder_id}):")
            drive_files = self.list_drive_files()
            
            if not drive_files:
                print("  ⚠️ No files found in the current folder")
                print("  This could mean:")
                print("  1. The folder is empty")
                print("  2. The folder ID is incorrect")
                print("  3. Permission issues")
                print("  4. Files are in a different folder")
            else:
                print(f"  Found {len(drive_files)} files:")
                for i, file_info in enumerate(drive_files[:20]):  # Show first 20
                    print(f"    {i+1}. {file_info['name']} (ID: {file_info['id']})")
                if len(drive_files) > 20:
                    print(f"    ... and {len(drive_files) - 20} more files")
            
            # Check for specific missing videos
            print(f"\n🎬 Checking for specific missing videos:")
            missing_videos = ['copay_health_insurance.mp4', 'acko_vs_star.mp4', 'care_supreme_health.mp4']
            
            for video_name in missing_videos:
                found = False
                for file_info in drive_files:
                    if file_info['name'] == video_name:
                        print(f"  ✅ Found: {video_name} (ID: {file_info['id']})")
                        found = True
                        break
                
                if not found:
                    print(f"  ❌ Not found: {video_name}")
                    # Look for similar names
                    similar = [f['name'] for f in drive_files if video_name.replace('.mp4', '').lower() in f['name'].lower()]
                    if similar:
                        print(f"    🔍 Similar files: {similar}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error debugging Google Drive setup: {e}")
            return False
    
    def transcribe_video(self, video_path: str) -> bool:
        """
        Transcribe a video file using Whisper.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"🎵 Transcribing: {os.path.basename(video_path)}")
            
            # Check file size and estimate time
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            print(f"📊 File size: {file_size:.1f} MB")
            
            # Estimate transcription time (rough estimate: 1 minute per 10MB)
            estimated_minutes = max(1, int(file_size / 10))
            print(f"⏱️ Estimated transcription time: ~{estimated_minutes} minutes")
            
            # Set timeout based on file size (minimum 5 minutes, maximum 30 minutes)
            timeout_minutes = min(30, max(5, estimated_minutes * 2))
            print(f"⏰ Timeout set to: {timeout_minutes} minutes")
            
            # Transcribe the video with progress indication
            print("🔄 Starting transcription (this may take several minutes)...")
            
            # Use a smaller model for faster transcription if available
            try:
                # Try to use a smaller model for faster processing
                if hasattr(self, 'whisper_model_small') and self.whisper_model_small is not None:
                    model = self.whisper_model_small
                    print("🎯 Using small model for faster transcription")
                else:
                    model = self.whisper_model
                    print("🎯 Using base model")
                
                # Transcribe with timeout
                import threading
                import queue
                
                result_queue = queue.Queue()
                error_queue = queue.Queue()
                
                def transcribe_worker():
                    try:
                        result = model.transcribe(video_path, verbose=True)
                        result_queue.put(result)
                    except Exception as e:
                        error_queue.put(e)
                
                # Start transcription in a separate thread
                transcribe_thread = threading.Thread(target=transcribe_worker)
                transcribe_thread.daemon = True
                transcribe_thread.start()
                
                # Wait for completion with timeout
                transcribe_thread.join(timeout=timeout_minutes * 60)
                
                if transcribe_thread.is_alive():
                    print(f"❌ Transcription timed out after {timeout_minutes} minutes")
                    return False
                
                # Check for errors
                if not error_queue.empty():
                    error = error_queue.get()
                    print(f"❌ Transcription failed: {error}")
                    return False
                
                # Get result
                if result_queue.empty():
                    print("❌ No transcription result received")
                    return False
                
                result = result_queue.get()
                print(f"✅ Transcription completed for: {os.path.basename(video_path)}")
                
            except Exception as whisper_error:
                print(f"❌ Whisper transcription failed: {whisper_error}")
                # Try with a different approach or fallback
                print("🔄 Attempting fallback transcription...")
                result = self.whisper_model.transcribe(video_path, verbose=True)
            
            # Save transcript as text file
            base_name = os.path.splitext(video_path)[0]
            txt_path = f"{base_name}.txt"
            
            print(f"💾 Saving transcript to: {os.path.basename(txt_path)}")
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                for segment in result['segments']:
                    start_time = segment['start']
                    end_time = segment['end']
                    text = segment['text'].strip()
                    f.write(f"[{start_time:.2f} - {end_time:.2f}] {text}\n")
            
            print(f"✅ Transcription saved: {os.path.basename(txt_path)}")
            print(f"📝 Generated {len(result['segments'])} segments")
            
            # Also generate SRT file
            print(f"📄 Generating SRT file...")
            self.generate_srt_file(video_path, result)
            
            return True
            
        except Exception as e:
            print(f"❌ Error transcribing {video_path}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_srt_file(self, video_path: str, result: dict = None) -> bool:
        """
        Generate SRT subtitle file from video transcription.
        
        Args:
            video_path: Path to the video file
            result: Whisper transcription result (if None, will transcribe)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            base_name = os.path.splitext(video_path)[0]
            srt_path = f"{base_name}.srt"
            
            # If no result provided, transcribe the video
            if result is None:
                result = self.whisper_model.transcribe(video_path)
            
            # Generate SRT file
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(result['segments'], 1):
                    start_time = self._format_srt_time(segment['start'])
                    end_time = self._format_srt_time(segment['end'])
                    text = segment['text'].strip()
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
            
            print(f"✅ SRT file saved: {os.path.basename(srt_path)}")
            return True
            
        except Exception as e:
            print(f"❌ Error generating SRT for {video_path}: {e}")
            return False

    def _format_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def process_new_files(self, files: List[str]):
        """
        Process new files by transcribing videos and uploading transcripts/SRTs.
        Only transcribe videos that don't already have transcript files.
        
        Args:
            files: List of new file names
        """
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        
        for file_name in files:
            file_path = os.path.join(self.local_sync_dir, file_name)
            file_ext = Path(file_name).suffix.lower()
            
            if file_ext in video_extensions:
                print(f"🎬 Processing video: {file_name}")
                
                # Check if transcript files already exist
                base_name = os.path.splitext(file_path)[0]
                txt_path = f"{base_name}.txt"
                srt_path = f"{base_name}.srt"
                
                txt_exists = os.path.exists(txt_path)
                srt_exists = os.path.exists(srt_path)
                
                # Check if transcript files exist in Google Drive as well
                drive_files = self.list_drive_files()
                drive_file_names = [f['name'] for f in drive_files]
                txt_name = os.path.basename(txt_path)
                srt_name = os.path.basename(srt_path)
                
                txt_in_drive = txt_name in drive_file_names
                srt_in_drive = srt_name in drive_file_names
                
                # Only transcribe if no transcript files exist locally or in Drive
                if not txt_exists and not srt_exists and not txt_in_drive and not srt_in_drive:
                    print(f"🎵 Transcribing video (no existing transcripts found): {file_name}")
                    
                    # Transcribe the video
                    if self.transcribe_video(file_path):
                        # Upload transcript to Google Drive
                        if os.path.exists(txt_path):
                            txt_name = os.path.basename(txt_path)
                            self.upload_file(txt_path, txt_name)
                            print(f"📤 Uploaded transcript: {txt_name}")
                        
                        # Generate and upload SRT file
                        if self.generate_srt_file(file_path, result=None):
                            srt_path = f"{base_name}.srt"
                            if os.path.exists(srt_path):
                                srt_name = os.path.basename(srt_path)
                                self.upload_file(srt_path, srt_name)
                                print(f"📤 Uploaded SRT: {srt_name}")
                else:
                    print(f"⏭️ Skipping transcription for {file_name} - transcript files already exist")
                    if txt_exists:
                        print(f"   📄 Local transcript: {txt_name}")
                    if srt_exists:
                        print(f"   📄 Local SRT: {srt_name}")
                    if txt_in_drive:
                        print(f"   ☁️ Drive transcript: {txt_name}")
                    if srt_in_drive:
                        print(f"   ☁️ Drive SRT: {srt_name}")
                    
                    # Upload existing local files to Drive if they're not there
                    if txt_exists and not txt_in_drive:
                        self.upload_file(txt_path, txt_name)
                        print(f"📤 Uploaded existing transcript: {txt_name}")
                    
                    if srt_exists and not srt_in_drive:
                        self.upload_file(srt_path, srt_name)
                        print(f"📤 Uploaded existing SRT: {srt_name}")
    
    def start_watching(self):
        """Start watching for new files in Google Drive."""
        if self.is_watching:
            print("⚠️ Already watching for changes")
            return
        
        self.is_watching = True
        self.watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.watch_thread.start()
        print(f"👀 Started watching Google Drive for changes (checking every {self.sync_interval}s)")
    
    def stop_watching(self):
        """Stop watching for new files."""
        self.is_watching = False
        if self.watch_thread:
            self.watch_thread.join()
        print("⏹️ Stopped watching Google Drive")
    
    def _watch_loop(self):
        """Main loop for watching Google Drive changes."""
        last_sync = {}
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        while self.is_watching:
            try:
                # Get current files from Google Drive
                drive_files = self.list_drive_files()
                current_files = {f['name']: f['modifiedTime'] for f in drive_files}
                
                # Check for new or modified files
                new_files = []
                for file_name, modified_time in current_files.items():
                    if file_name not in last_sync or last_sync[file_name] != modified_time:
                        new_files.append(file_name)
                
                if new_files:
                    print(f"🔄 Found {len(new_files)} new/modified files: {new_files}")
                    
                    # Sync files from Google Drive
                    sync_result = self.sync_from_drive()
                    
                    # Process new files (transcribe videos)
                    all_new = sync_result['downloaded'] + sync_result['updated']
                    if all_new:
                        self.process_new_files(all_new)
                    
                    # Update last sync times
                    for file_name in new_files:
                        if file_name in current_files:
                            last_sync[file_name] = current_files[file_name]
                    
                    # Reset error counter on successful sync
                    consecutive_errors = 0
                
                # Wait before next check
                time.sleep(self.sync_interval)
                
            except Exception as e:
                consecutive_errors += 1
                print(f"❌ Error in watch loop (attempt {consecutive_errors}/{max_consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    print(f"⚠️ Too many consecutive errors ({consecutive_errors}), pausing sync for 60 seconds...")
                    time.sleep(60)
                    consecutive_errors = 0
                else:
                    # Exponential backoff for errors
                    backoff_time = min(self.sync_interval * (2 ** consecutive_errors), 60)
                    print(f"⏳ Waiting {backoff_time} seconds before retry...")
                    time.sleep(backoff_time)
    
    def check_and_transcribe_existing_videos(self) -> Dict[str, List[str]]:
        """
        Check for existing videos in the local folder that don't have transcript files
        and transcribe them automatically.
        
        Returns:
            Dictionary with 'transcribed', 'skipped', and 'failed' lists
        """
        transcribed = []
        skipped = []
        failed = []
        
        if not os.path.exists(self.local_sync_dir):
            return {'transcribed': transcribed, 'skipped': skipped, 'failed': failed}
        
        print("🎵 Checking for existing videos that need transcription...")
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        
        for file_name in os.listdir(self.local_sync_dir):
            file_ext = Path(file_name).suffix.lower()
            
            if file_ext in video_extensions:
                file_path = os.path.join(self.local_sync_dir, file_name)
                base_name = os.path.splitext(file_path)[0]
                txt_path = f"{base_name}.txt"
                srt_path = f"{base_name}.srt"
                
                txt_exists = os.path.exists(txt_path)
                srt_exists = os.path.exists(srt_path)
                
                # Check if transcript files exist in Google Drive as well
                drive_files = self.list_drive_files()
                drive_file_names = [f['name'] for f in drive_files]
                txt_name = os.path.basename(txt_path)
                srt_name = os.path.basename(srt_path)
                
                txt_in_drive = txt_name in drive_file_names
                srt_in_drive = srt_name in drive_file_names
                
                # Only transcribe if no transcript files exist locally or in Drive
                if not txt_exists and not srt_exists and not txt_in_drive and not srt_in_drive:
                    print(f"🎵 Transcribing existing video: {file_name}")
                    
                    try:
                        # Transcribe the video
                        if self.transcribe_video(file_path):
                            transcribed.append(file_name)
                            
                            # Upload transcript to Google Drive
                            if os.path.exists(txt_path):
                                txt_name = os.path.basename(txt_path)
                                self.upload_file(txt_path, txt_name)
                                print(f"📤 Uploaded transcript: {txt_name}")
                            
                            # Generate and upload SRT file
                            if self.generate_srt_file(file_path, result=None):
                                srt_path = f"{base_name}.srt"
                                if os.path.exists(srt_path):
                                    srt_name = os.path.basename(srt_path)
                                    self.upload_file(srt_path, srt_name)
                                    print(f"📤 Uploaded SRT: {srt_name}")
                        else:
                            failed.append(file_name)
                            print(f"❌ Failed to transcribe: {file_name}")
                    except Exception as e:
                        failed.append(file_name)
                        print(f"❌ Error transcribing {file_name}: {e}")
                else:
                    skipped.append(file_name)
                    print(f"⏭️ Skipping {file_name} - transcript files already exist")
                    if txt_exists:
                        print(f"   📄 Local transcript: {txt_name}")
                    if srt_exists:
                        print(f"   📄 Local SRT: {srt_name}")
                    if txt_in_drive:
                        print(f"   ☁️ Drive transcript: {txt_name}")
                    if srt_in_drive:
                        print(f"   ☁️ Drive SRT: {srt_name}")
                    
                    # Upload existing local files to Drive if they're not there
                    if txt_exists and not txt_in_drive:
                        self.upload_file(txt_path, txt_name)
                        print(f"📤 Uploaded existing transcript: {txt_name}")
                    
                    if srt_exists and not srt_in_drive:
                        self.upload_file(srt_path, srt_name)
                        print(f"📤 Uploaded existing SRT: {srt_name}")
        
        if transcribed:
            print(f"✅ Transcribed {len(transcribed)} videos: {transcribed}")
        if skipped:
            print(f"⏭️ Skipped {len(skipped)} videos (already have transcripts)")
        if failed:
            print(f"❌ Failed to transcribe {len(failed)} videos: {failed}")
        
        return {
            'transcribed': transcribed,
            'skipped': skipped,
            'failed': failed
        }

    def initial_sync(self):
        """Perform initial sync from Google Drive."""
        print("🔄 Performing initial sync from Google Drive...")
        
        # Clean up any interrupted sync operations
        self._cleanup_interrupted_sync()
        
        sync_result = self.sync_from_drive()
        
        print(f"📥 Downloaded: {len(sync_result['downloaded'])} files")
        print(f"🔄 Updated: {len(sync_result['updated'])} files")
        if sync_result['errors']:
            print(f"❌ Errors: {len(sync_result['errors'])} files")
        
        # Report on missing video downloads
        if 'missing_videos_downloaded' in sync_result and sync_result['missing_videos_downloaded']:
            print(f"🎬 Downloaded {len(sync_result['missing_videos_downloaded'])} missing video files")
        if 'missing_videos_failed' in sync_result and sync_result['missing_videos_failed']:
            print(f"❌ Failed to download {len(sync_result['missing_videos_failed'])} missing video files")
        if 'missing_videos_not_found' in sync_result and sync_result['missing_videos_not_found']:
            print(f"⚠️ {len(sync_result['missing_videos_not_found'])} video files not found in Google Drive")
        
        # Process any new files
        all_new = sync_result['downloaded'] + sync_result['updated']
        if all_new:
            self.process_new_files(all_new)
        
        # Check and transcribe any existing videos that don't have transcripts
        transcription_result = self.check_and_transcribe_existing_videos()
        
        # Add transcription results to sync result
        sync_result['transcribed'] = transcription_result['transcribed']
        sync_result['transcription_skipped'] = transcription_result['skipped']
        sync_result['transcription_failed'] = transcription_result['failed']
        
        return sync_result

    def _cleanup_interrupted_sync(self):
        """Clean up any temporary files from interrupted sync operations."""
        if not os.path.exists(self.local_sync_dir):
            return
        
        print("🧹 Cleaning up interrupted sync operations...")
        cleaned_count = 0
        
        for file_name in os.listdir(self.local_sync_dir):
            if file_name.endswith('.tmp'):
                temp_path = os.path.join(self.local_sync_dir, file_name)
                try:
                    os.remove(temp_path)
                    cleaned_count += 1
                    print(f"🗑️ Removed temp file: {file_name}")
                except Exception as e:
                    print(f"⚠️ Could not remove temp file {file_name}: {e}")
        
        if cleaned_count > 0:
            print(f"✅ Cleaned up {cleaned_count} temporary files")
        else:
            print("✅ No temporary files found")

    def sync_folder(self) -> Dict[str, List[str]]:
        """
        Sync the entire folder from Google Drive.
        This is a convenience method that combines cleanup and sync.
        
        Returns:
            Dictionary with sync results
        """
        print("🔄 Starting folder sync...")
        
        # Clean up any interrupted operations
        self._cleanup_interrupted_sync()
        
        # Perform sync
        sync_result = self.sync_from_drive()
        
        # Print summary
        print(f"\n📊 Sync Summary:")
        print(f"📥 Downloaded: {len(sync_result['downloaded'])} files")
        print(f"🔄 Updated: {len(sync_result['updated'])} files")
        if sync_result['errors']:
            print(f"❌ Errors: {len(sync_result['errors'])} files")
        
        # Report on missing video downloads
        if 'missing_videos_downloaded' in sync_result and sync_result['missing_videos_downloaded']:
            print(f"🎬 Downloaded {len(sync_result['missing_videos_downloaded'])} missing video files")
        if 'missing_videos_failed' in sync_result and sync_result['missing_videos_failed']:
            print(f"❌ Failed to download {len(sync_result['missing_videos_failed'])} missing video files")
        if 'missing_videos_not_found' in sync_result and sync_result['missing_videos_not_found']:
            print(f"⚠️ {len(sync_result['missing_videos_not_found'])} video files not found in Google Drive")
        
        # Report on transcription results
        if 'transcribed' in sync_result and sync_result['transcribed']:
            print(f"🎵 Transcribed {len(sync_result['transcribed'])} videos: {sync_result['transcribed']}")
        if 'transcription_skipped' in sync_result and sync_result['transcription_skipped']:
            print(f"⏭️ Skipped transcription for {len(sync_result['transcription_skipped'])} videos (already have transcripts)")
        if 'transcription_failed' in sync_result and sync_result['transcription_failed']:
            print(f"❌ Failed to transcribe {len(sync_result['transcription_failed'])} videos: {sync_result['transcription_failed']}")
        
        return sync_result

    def transcribe_existing_videos_only(self) -> Dict[str, List[str]]:
        """
        Only transcribe existing videos that don't have transcript files.
        This is useful when you just want to transcribe videos without doing a full sync.
        
        Returns:
            Dictionary with transcription results
        """
        print("🎵 Transcribing existing videos only...")
        return self.check_and_transcribe_existing_videos()

# Example usage and setup
def setup_google_drive_sync():
    """
    Setup function to initialize Google Drive sync.
    Call this before using the video processing pipeline.
    """
    try:
        # Initialize Google Drive sync
        drive_sync = GoogleDriveSync()
        
        # Perform initial sync
        drive_sync.initial_sync()
        
        # Start watching for changes
        drive_sync.start_watching()
        
        return drive_sync
        
    except Exception as e:
        print(f"❌ Failed to setup Google Drive sync: {e}")
        return None

if __name__ == "__main__":
    print("=== Google Drive Sync for ClipQuery ===")
    print("This script will help you sync your video transcripts from Google Drive.")
    print("You can choose an existing folder or create a new 'Max Life Videos' folder.")
    print()
    
    # Initialize the enhanced vector store and start the sync process
    vector_store = VectorStore()
    vector_store.load_transcripts()
    
    print("\n=== Sync Complete ===")
    print("Your video transcripts have been downloaded and processed.")
    print("You can now use your ClipQuery application with the synced data.")
    print("\nTo stop the sync process, press Ctrl+C")
    
    # Keep the script running to maintain the sync
    try:
        while True:
            import time
            time.sleep(60)  # Check for updates every minute
            print("Checking for new files...")
            # You could add logic here to check for new files periodically
    except KeyboardInterrupt:
        print("\nSync stopped by user.") 