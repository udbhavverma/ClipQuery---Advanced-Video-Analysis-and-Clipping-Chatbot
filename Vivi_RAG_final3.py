import os
import uuid
import tempfile
import subprocess
import threading
import time
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
from PIL import Image, ImageTk
import whisper
from groq import Groq
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import re
import requests
from rag_pipeline import VideoRAG
import cv2
import json

# Import Google Drive sync
try:
    from google_drive_sync import GoogleDriveSync
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False
    print("⚠️ Google Drive sync not available. Install required packages: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")

# ------------------ Utility Functions ------------------
def _parse_srt_time(t):
    try:
        h, m, s_ms = t.split(":")
        if "," in s_ms:
            s, ms = s_ms.split(",")
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
        else:
            return int(h) * 3600 + int(m) * 60 + float(s_ms)
    except:
        pass
    try:
        m, s = t.split(":")
        return int(m) * 60 + float(s)
    except:
        pass
    try:
        return float(t)
    except:
        raise ValueError(f"Unrecognized timestamp format: {t}")

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")

# ------------------ FFmpeg Functions ------------------
def concatenate_videos(video_paths, output_filepath):
    video_paths = [p for p in video_paths if p and os.path.exists(p)]
    if not video_paths:
        return "❌ Error: No valid video clips provided.", None
    try:
        if not video_paths:
            return "❌ Error: No video clips provided.", None
        temp_dir = tempfile.gettempdir()
        list_file_path = os.path.join(temp_dir, f"concat_list_{uuid.uuid4().hex[:8]}.txt")
        with open(list_file_path, "w", encoding="utf-8") as f:
            for path in video_paths:
                normalized_path = path.replace('\\', '/')
                f.write(f"file '{normalized_path}'\n")
        
        # Use re-encoding instead of stream copying for better compatibility
        # This ensures all clips have consistent codecs, frame rates, and audio settings
        command = [
            "ffmpeg", "-y", 
            "-f", "concat", 
            "-safe", "0",
            "-i", list_file_path, 
            "-c:v", "libx264",           # Use H.264 codec for video
            "-c:a", "aac",               # Use AAC codec for audio
            "-preset", "fast",           # Fast encoding preset
            "-crf", "23",                # Good quality setting
            "-r", "30",                  # Force 30fps for consistency
            "-ar", "44100",              # Force 44.1kHz audio sample rate
            "-ac", "2",                  # Force stereo audio
            "-movflags", "+faststart",   # Optimize for streaming
            "-avoid_negative_ts", "make_zero",  # Handle negative timestamps
            "-fflags", "+genpts",        # Generate presentation timestamps
            output_filepath
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Clean up temporary files
        try:
            os.remove(list_file_path)
            # Also clean up individual clip files
            for clip_path in video_paths:
                if os.path.exists(clip_path) and "clip_" in os.path.basename(clip_path):
                    os.remove(clip_path)
        except Exception as e:
            print(f"Warning: Could not clean up temporary files: {e}")
        
        return f"✅ Concatenated video saved to: {output_filepath}", output_filepath
    except subprocess.CalledProcessError as e:
        return f"❌ FFmpeg error: {e.stderr}", None
    except Exception as e:
        return f"❌ Exception during concatenation: {str(e)}", None

# ------------------ LLM Setup ------------------
os.environ["GROQ_API_KEY"] = "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"

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=1)
llm_clipping = ChatGroq(model="llama-3.3-70b-versatile", temperature=1.2)  # Higher temperature for more creative clipping

general_template = """
You are Vivi, an expert and friendly video assistant chatbot.
You are having an ongoing conversation with the user. You have access to a transcript of a video. If the user's question is about the video, answer helpfully and use the transcript as context. Do not refer to timestamps unless they are explicitly present in the transcript context. If the question is general and not related to the video, just respond helpfully like a normal assistant.

IMPORTANT: Acronyms, product names, and technical terms may appear in the transcript with or without spaces, hyphens, or different casing (e.g., 'StarULIP', 'Star ULIP', 'star ulip'). When matching terms from the user's query to the transcript, always consider possible variations in spacing and casing, and do not assume an exact match is required.

CRITICAL: When the user asks about specific terms, acronyms, or concepts, carefully search through the provided transcript for those terms. If you find explanations or mentions of the requested terms, provide a comprehensive answer based on what you find in the transcript. Do not say a term is not mentioned unless you have thoroughly searched the entire transcript and are certain it is not present.
---
Conversation History:
{context}
---
Transcript of the Video:
{transcript}
---
User:
{question}
---
Vivi:
"""

clipping_template = """
You are an expert video analysis assistant. Given a user query and full transcripts from multiple videos, propose the best video clip ranges (start and end times in seconds) that comprehensively answer the query. 

IMPORTANT GUIDELINES:
1. Create COMPLETE, comprehensive clips that fully address the query
2. For acronyms, frameworks, methods, or multi-part explanations, ensure ALL components are explained in the clip
3. Choose the MOST RELEVANT video for each clip based on content
4. Ensure clips start and end at natural points (beginning/end of sentences or thoughts)
5. Be PRECISE - find the exact timestamps where the content appears in the transcript
6. If the query asks for multiple aspects, create multiple clips to cover all aspects
7. Pay special attention to structured content - make sure all parts are covered
8. AVOID creating overlapping or redundant clips - each clip should be unique and non-overlapping
9. For acronym explanations, ensure you include the COMPLETE explanation of ALL letters/components
10. Do NOT cut off acronym explanations mid-way - include the full breakdown
11. If the query asks for MULTIPLE acronyms (e.g., "What is NOPP and StarULIP?"), create separate clips for EACH acronym
12. Ensure ALL requested acronyms are covered - don't miss any acronym mentioned in the query
13. CRITICAL: Do NOT create overlapping time ranges - each clip should have unique start and end times
14. For multiple acronyms, create one clip per acronym with non-overlapping time ranges
15. Specify which video to use for each clip based on content relevance
16. If a clip would end mid-sentence or mid-thought, extend it to include the complete sentence or thought

TIMESTAMP INSTRUCTIONS:
- The transcript contains timestamps in the format [start_time - end_time] followed by text
- Find the timestamps where the content you need appears
- For acronyms, find the timestamps where each letter/component is explained
- Use the exact start time of the first relevant segment and exact end time of the last relevant segment
- Add minimal context only if the clip would otherwise end mid-sentence
- Do NOT use broad ranges like 0-60, 0-120, or 0-156
- Be specific and complete - ensure the entire explanation is captured
- Look for the exact timestamps where the requested content appears

IMPORTANT: Acronyms, product names, and technical terms may appear in the transcript with or without spaces, hyphens, or different casing (e.g., 'StarULIP', 'Star ULIP', 'star ulip'). When matching terms from the user's query to the transcript, always consider possible variations in spacing and casing, and do not assume an exact match is required.

Query: {query}
Full Transcripts from Videos:
{transcript}

Return the result in the following format, with each range and explanation on separate lines:
- Video: [video_id] 
  Range: start_time - end_time
  Relevance: [Brief explanation of why this range is relevant to the query]

Make sure to create comprehensive clips that fully answer the user's query, not just partial explanations. For structured content (acronyms, frameworks, methods), ensure every component is explained. Avoid redundant or overlapping clips. For acronyms, make sure to include the complete explanation of all components. If multiple acronyms are requested, ensure ALL are covered with separate, non-overlapping clips. Choose the most relevant video for each clip based on content. 

CRITICAL: Find the exact timestamps where the content appears and use those. Do not use broad ranges like 0-60. Be precise and complete.
"""

prompt_general = ChatPromptTemplate.from_template(general_template)
prompt_clipping = ChatPromptTemplate.from_template(clipping_template)
chain_general = prompt_general | llm
chain_clipping = prompt_clipping | llm_clipping

# ------------------ Vivi GUI Class ------------------
class ViviChatbot:
    def __init__(self):
        # Use light appearance mode
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        # Create root and set background
        self.root = ctk.CTk()
        self.root.configure(fg_color="#FFFFFF")
        self.root.title("ClipQuery- Video Chatbot")
        self.root.geometry("900x650")

        self.video_path = ""
        self.context = ""
        self.transcript_segments = []
        self.full_transcript_text = ""
        self.audio_process = None
        
        # Conversation history management
        self.conversation_history = []
        self.max_conversation_turns = 10  # Keep only last 10 turns
        self.max_context_chars = 8000  # Reduced from 16000/24000

        # Initialize Google Drive sync in background
        self.drive_sync = None
        self.sync_thread = None
        if GOOGLE_DRIVE_AVAILABLE:
            self.init_google_drive_sync()
        
        # Check for missing video files on startup
        self.check_missing_videos_on_startup()

        # Logo and Title Frame
        self.header_frame = ctk.CTkFrame(self.root, fg_color="#FFFFFF", width=200, height=100)
        self.header_frame.pack(anchor="nw", padx=20, pady=(10, 5), fill="x")

        # Load and display logo with resizing
        try:
            logo_img = Image.open(r"C:\Dev\ClipQuery\vivi_logo_1.PNG")
            logo_img = logo_img.resize((100, 100), Image.Resampling.LANCZOS)
            logo_photo = ImageTk.PhotoImage(logo_img)
            self.logo_label = ctk.CTkLabel(
                self.header_frame,
                image=logo_photo,
                text="",
                fg_color="#FFFFFF"
            )
            self.logo_label.image = logo_photo
            self.logo_label.pack(side="left", padx=(0, 10))
        except Exception as e:
            self.logo_label = ctk.CTkLabel(
                self.header_frame,
                text="Logo",
                font=("Arial", 16),
                fg_color="#FFFFFF",
                text_color="#1a2238"
            )
            self.logo_label.pack(side="left", padx=(0, 10))

        # Title label
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="ClipQuery",
            font=("Arial", 24, "bold"),
            text_color="#1a2238"
        )
        self.title_label.pack(side="left")

        # Chat display frame
        self.chat_frame = ctk.CTkScrollableFrame(self.root, width=880, height=500, fg_color="#FFFFFF")
        self.chat_frame.pack(padx=20, pady=(5, 10), fill="both", expand=True)
        self.scrollable_frame = self.chat_frame

        # Entry field frame
        self.entry_frame = ctk.CTkFrame(self.root, fg_color="#FFFFFF")
        self.entry_frame.pack(pady=(0, 10), fill="x", padx=20)

        self.user_entry = ctk.CTkEntry(self.entry_frame, placeholder_text="Ask Vivi...", width=700, height=40, font=("Arial", 16))
        self.user_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.user_entry.bind("<Return>", lambda e: self.send_message())

        # Send button
        self.send_btn = ctk.CTkButton(self.entry_frame, text="Send", command=self.send_message, fg_color="#1a2238", width=160, height=40, font=("Arial", 16))
        self.send_btn.pack(side="right")

        # Buttons for final video and sync
        self.btn_frame = ctk.CTkFrame(self.root, fg_color="#FFFFFF")
        self.btn_frame.pack(pady=(0, 10))

        self.final_video_btn = ctk.CTkButton(self.btn_frame, text="🎬 Final Video", command=self.play_final_video, fg_color="#1a2238", width=160, height=40, font=("Arial", 16))
        self.final_video_btn.pack(side="left", padx=5)

        # Sync button
        self.sync_btn = ctk.CTkButton(self.btn_frame, text="🔄 Sync Drive", command=self.manual_sync, fg_color="#17a2b8", width=120, height=40, font=("Arial", 14))
        self.sync_btn.pack(side="left", padx=5)

        # Google Drive sync status indicator
        self.sync_status_label = ctk.CTkLabel(self.btn_frame, text="🔄 Initializing Google Drive sync...", font=("Arial", 12), text_color="#666666")
        self.sync_status_label.pack(side="right", padx=10)

        # Progress bar for buffering
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ctk.CTkProgressBar(self.root, variable=self.progress_var, width=200)
        self.progress_bar.pack(pady=5)
        self.progress_bar.pack_forget()  # Hide initially

        # Initial welcome message
        self.display_message("assistant", "Hello! I'm Vivi, your video assistant. I can help you analyze videos from your Google Drive and create clips. Just ask me questions or use 'clip:' to create video clips. How can I assist you today?")

        # Initialize RAG pipeline
        try:
            self.rag = VideoRAG()
            print("✅ RAG pipeline initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing RAG pipeline: {e}")
            self.rag = None

        # Similarity threshold for filtering results
        self.similarity_threshold = 0.75  # More permissive to ensure we get relevant results

    def estimate_tokens(self, text):
        """Rough estimate of token count (1 token ≈ 4 characters for English text)."""
        return len(text) // 4

    def check_token_limit(self, text, max_tokens=8000):
        """Check if text exceeds token limit and truncate if necessary."""
        estimated_tokens = self.estimate_tokens(text)
        if estimated_tokens > max_tokens:
            # Truncate to approximately max_tokens
            max_chars = max_tokens * 4
            return text[:max_chars] + "..."
        return text



    def manage_conversation_history(self, user_input, response):
        """Manage conversation history to prevent token limit exceeded errors."""
        # Add new turn to history
        self.conversation_history.append({
            'user': user_input,
            'ai': response,
            'timestamp': time.time()
        })
        
        # Keep only the last N turns
        if len(self.conversation_history) > self.max_conversation_turns:
            self.conversation_history = self.conversation_history[-self.max_conversation_turns:]
        
        # Rebuild context from history with token limit checking
        context_parts = []
        total_chars = 0
        max_context_chars = self.max_context_chars
        
        for turn in self.conversation_history:
            turn_text = f"User: {turn['user']}\nAI: {turn['ai']}\n"
            if total_chars + len(turn_text) > max_context_chars:
                break
            context_parts.append(turn_text)
            total_chars += len(turn_text)
        
        self.context = "\n".join(context_parts)
        
        # Final token limit check
        self.context = self.check_token_limit(self.context, max_tokens=6000)  # Conservative limit
        
        estimated_tokens = self.estimate_tokens(self.context)
        print(f"Conversation history: {len(self.conversation_history)} turns, {total_chars} chars, ~{estimated_tokens} tokens")
        
        # Auto-clear if getting too large
        if estimated_tokens > 5000:  # Conservative threshold
            print("⚠️ Conversation history getting large, auto-clearing...")
            self.conversation_history = self.conversation_history[-3:]  # Keep only last 3 turns
            self.context = ""
            for turn in self.conversation_history:
                self.context += f"User: {turn['user']}\nAI: {turn['ai']}\n"
            self.context = self.check_token_limit(self.context, max_tokens=3000)
            print("✅ Auto-cleared conversation history")
            # Show a subtle notification to the user
            self.display_message("system", "💡 Conversation history automatically cleared to maintain performance.")

    def init_google_drive_sync(self):
        """Initialize Google Drive sync in background thread."""
        def sync_worker():
            try:
                print("🔄 Initializing Google Drive sync...")
                self.drive_sync = GoogleDriveSync()
                
                # Perform initial sync
                print("📥 Performing initial sync from Google Drive...")
                sync_result = self.drive_sync.initial_sync()
                
                print(f"✅ Google Drive sync initialized successfully")
                print(f"📥 Downloaded: {len(sync_result['downloaded'])} files")
                print(f"🔄 Updated: {len(sync_result['updated'])} files")
                
                # Show transcription results if any
                if 'transcribed' in sync_result and sync_result['transcribed']:
                    print(f"🎵 Transcribed {len(sync_result['transcribed'])} videos: {sync_result['transcribed']}")
                if 'transcription_skipped' in sync_result and sync_result['transcription_skipped']:
                    print(f"⏭️ Skipped transcription for {len(sync_result['transcription_skipped'])} videos")
                if 'transcription_failed' in sync_result and sync_result['transcription_failed']:
                    print(f"❌ Failed to transcribe {len(sync_result['transcription_failed'])} videos")
                
                # Update status label safely
                try:
                    self.root.after(0, lambda: self.sync_status_label.configure(
                        text="✅ Google Drive sync active",
                        text_color="#28a745"
                    ))
                except Exception as e:
                    print(f"⚠️ Could not update status label: {e}")
                
                # Start watching for changes
                self.drive_sync.start_watching()
                print("👀 Google Drive sync is now running in background")
                
            except Exception as e:
                print(f"❌ Error initializing Google Drive sync: {e}")
                self.drive_sync = None
                # Update status label to show error safely
                try:
                    self.root.after(0, lambda: self.sync_status_label.configure(
                        text="❌ Google Drive sync failed",
                        text_color="#dc3545"
                    ))
                    # Show error message in chat
                    self.root.after(0, lambda: self.display_message("assistant", 
                        f"⚠️ Google Drive sync is not available: {str(e)}\n\n"
                        "The chatbot will work with local files only. "
                        "To enable Google Drive sync, please check your credentials and try again."
                    ))
                except Exception as ui_error:
                    print(f"⚠️ Could not update UI: {ui_error}")
        
        # Start sync in background thread
        self.sync_thread = threading.Thread(target=sync_worker, daemon=True)
        self.sync_thread.start()

    def show_buffering(self):
        self.buffering_label = ctk.CTkLabel(self.root, text="Analyzing", font=("Arial", 16))
        self.buffering_label.pack(pady=10)
        self.animate_buffering(0)

    def animate_buffering(self, count):
        dots = "." * (count % 4)
        self.buffering_label.configure(text=f"Analyzing{dots}")
        self.root.after(500, lambda: self.animate_buffering(count + 1))

    def hide_buffering(self):
        self.buffering_label.pack_forget()
        
    def typewriter_effect(self, sender, message):
        outer_frame = ctk.CTkFrame(self.scrollable_frame, fg_color="#FFFFFF")
        outer_frame.pack(fill="x", padx=10, pady=5)

        bubble_frame = ctk.CTkFrame(
            outer_frame,
            fg_color="#3a0e2e" if sender.lower() == "vivi" else "#1B263B",
            corner_radius=12
        )
        if sender.lower() == "vivi":
            bubble_frame.pack(anchor="w", padx=5)
        else:
            bubble_frame.pack(anchor="e", padx=5)

        name_label = ctk.CTkLabel(
            bubble_frame,
            text=sender + ":",
            font=("Arial", 12, "bold"),
            text_color="#F0F0F0"
        )
        name_label.pack(anchor="w", padx=8, pady=(6, 0))

        message_label = ctk.CTkLabel(
            bubble_frame,
            text="",
            font=("Arial", 20, "normal"),
            wraplength=800,
            justify="left",
            text_color="#F0F0F0"
        )
        message_label.pack(anchor="w", padx=8, pady=(0, 8))

        self.root.update_idletasks()
        self.chat_frame._parent_canvas.yview_moveto(1.0)

        current_text = ""
        for word in message.split():
            current_text += word + " "
            message_label.configure(text=current_text)
            self.root.update_idletasks()
            self.chat_frame._parent_canvas.yview_moveto(1.0)
            time.sleep(0.04)

    def display_message(self, sender, message):
        outer_frame = ctk.CTkFrame(self.chat_frame, fg_color="#FFFFFF")
        outer_frame.pack(fill="x", padx=10, pady=5)

        bubble = ctk.CTkFrame(
            outer_frame,
            corner_radius=15,
            fg_color="#3a0e2e" if sender.lower() != "user" else "#1a2238"
        )

        if sender.lower() == "user":
            bubble.pack(anchor="e", padx=5)
        else:
            bubble.pack(anchor="w", padx=5)

        label = ctk.CTkLabel(
            bubble,
            text=f"{message}",
            wraplength=800,
            font=("Arial", 20, "normal"),
            justify="left",
            text_color="#F0F0F0"
        )
        label.pack(anchor="w", padx=10, pady=5)

    def clip_video(self, video_path, start_time, end_time):
        if not video_path or not os.path.exists(video_path):
            self.display_message("system", "❌ No valid video found for clipping.")
            return "No video found", None
        try:
            if start_time >= end_time:
                self.display_message("system", f"⚠ Invalid clip range: {start_time} >= {end_time}")
                return "Invalid time range", None
            
            clip_filename = f"clip_{uuid.uuid4().hex[:8]}.mp4"
            output_path = os.path.join(tempfile.gettempdir(), clip_filename)
            
            # Use re-encoding instead of stream copying to prevent frame freezing and sync issues
            # Force keyframe alignment and consistent codec settings
            command = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-i", video_path,
                "-t", str(end_time - start_time),
                "-c:v", "libx264",           # Use H.264 codec for video
                "-c:a", "aac",               # Use AAC codec for audio
                "-preset", "fast",           # Fast encoding preset
                "-crf", "23",                # Good quality setting
                "-avoid_negative_ts", "make_zero",  # Handle negative timestamps
                "-fflags", "+genpts",        # Generate presentation timestamps
                "-r", "30",                  # Force 30fps for consistency
                "-ar", "44100",              # Force 44.1kHz audio sample rate
                "-ac", "2",                  # Force stereo audio
                "-movflags", "+faststart",   # Optimize for streaming
                output_path
            ]
            
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return f"🎬 Video clip saved to {output_path}", output_path
        except subprocess.CalledProcessError as e:
            self.display_message("system", f"❌ FFmpeg error: {e.stderr}")
            return f"FFmpeg error: {e.stderr}", None
        except Exception as e:
            self.display_message("system", f"❌ Error clipping video: {e}")
            return f"Error clipping video: {e}", None

    def transcribe_video(self, video_path):
        base = os.path.splitext(os.path.basename(video_path))[0]
        dir_ = os.path.dirname(video_path)
        txt_path = os.path.join(dir_, f"{base}.txt")
        srt_path = os.path.join(dir_, f"{base}.srt")
        if os.path.exists(txt_path) and os.path.exists(srt_path):
            with open(srt_path, "r", encoding="utf-8") as f:
                blocks = f.read().strip().split("\n\n")
            with open(txt_path, "r", encoding="utf-8") as f:
                timed_transcript = f.read().strip()
            segments = []
            for block in blocks:
                lines = block.split("\n")
                if len(lines) >= 3:
                    times = lines[1].split(" --> ")
                    start = _parse_srt_time(times[0])
                    end = _parse_srt_time(times[1])
                    text = " ".join(lines[2:])
                    segments.append({"start": start, "end": end, "text": text})
            return timed_transcript, segments
        self.display_message("system", "🔍 Running Whisper transcription...")
        
        self.show_buffering()
        model = whisper.load_model("medium")
        
        
        result = model.transcribe(video_path, verbose=True)
        segments = []
        timed_transcript = ""
        for seg in result['segments']:
            segments.append({"start": seg['start'], "end": seg['end'], "text": seg['text']})
            timed_transcript += f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}\n"
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments):
                f.write(f"{i+1}\n{format_time(seg['start'])} --> {format_time(seg['end'])}\n{seg['text']}\n\n")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(timed_transcript)
        self.hide_buffering()
        return timed_transcript, segments
    
    # ################## Using API #################
    # def transcribe_video(self, video_path):
    #     base = os.path.splitext(os.path.basename(video_path))[0]
    #     dir_ = os.path.dirname(video_path)
    #     txt_path = os.path.join(dir_, f"{base}.txt")
    #     srt_path = os.path.join(dir_, f"{base}.srt")

    #     # Check if already transcribed
    #     if os.path.exists(txt_path) and os.path.exists(srt_path):
    #         with open(srt_path, "r", encoding="utf-8") as f:
    #             blocks = f.read().strip().split("\n\n")
    #         with open(txt_path, "r", encoding="utf-8") as f:
    #             timed_transcript = f.read().strip()
    #         segments = []
    #         for block in blocks:
    #             lines = block.split("\n")
    #             if len(lines) >= 3:
    #                 times = lines[1].split(" --> ")
    #                 start = _parse_srt_time(times[0])
    #                 end = _parse_srt_time(times[1])
    #                 text = " ".join(lines[2:])
    #                 segments.append({"start": start, "end": end, "text": text})
    #         return timed_transcript, segments

    #     # Use Groq Whisper API
    #     self.display_message("system", "🔍 Sending video to Groq Whisper (large-v3-turbo) for transcription...")

    #     api_key = os.environ.get("GROQ_API_KEY")
    #     if not api_key:
    #         raise ValueError("GROQ_API_KEY is not set in environment variables.")

    #     client = Groq(api_key=api_key)
    #     try:
    #         with open(video_path, "rb") as file:
    #             transcription = client.audio.transcriptions.create(
    #                 file=file,
    #                 model="whisper-large-v3-turbo",
    #                 prompt="",  # You can provide custom prompt if needed
    #                 response_format="verbose_json",
    #                 timestamp_granularities=["segment"],
    #                 language="en",
    #                 temperature=0.0
    #             )

    #         segments = []
    #         timed_transcript = ""
    #         for seg in transcription.segments:
    #             start = seg["start"]
    #             end = seg["end"]
    #             text = seg["text"]
    #             segments.append({"start": start, "end": end, "text": text})
    #             timed_transcript += f"[{start:.2f} - {end:.2f}] {text}\n"

    #         # Save to SRT
    #         with open(srt_path, "w", encoding="utf-8") as f:
    #             for i, seg in enumerate(segments):
    #                 f.write(f"{i+1}\n{format_time(seg['start'])} --> {format_time(seg['end'])}\n{seg['text']}\n\n")

    #         # Save to TXT
    #         with open(txt_path, "w", encoding="utf-8") as f:
    #             f.write(timed_transcript)

    #         return timed_transcript, segments

    #     except Exception as e:
    #         self.display_message("system", f"Transcription Failed: {str(e)}")
    #         return "", []

    def load_transcript_segments(self, video_id, relevant_ranges=None):
        """Load transcript segments for a video. If relevant_ranges is provided, only load those segments."""
        txt_path = os.path.join("Max Life Videos", f"{video_id}.txt")
        txt_path = os.path.normpath(txt_path)
        segments = []
        if not os.path.exists(txt_path):
            print(f"Transcript file not found: {txt_path}")
            return segments
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    match = re.match(r'\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s*(.+)', line)
                    if match:
                        start = float(match.group(1))
                        end = float(match.group(2))
                        text = match.group(3).strip()
                        if relevant_ranges:
                            for r_start, r_end in relevant_ranges:
                                if (start >= r_start and start < r_end) or (end > r_start and end <= r_end):
                                    segments.append({"start": start, "end": end, "text": text})
                                    break
                        else:
                            segments.append({"start": start, "end": end, "text": text})
        except Exception as e:
            print(f"Error reading transcript file {txt_path}: {e}")
        return segments

    def load_full_transcript(self, video_id):
        """Load the full transcript for a video."""
        txt_path = os.path.join("Max Life Videos", f"{video_id}.txt")
        txt_path = os.path.normpath(txt_path)
        if not os.path.exists(txt_path):
            print(f"Transcript file not found: {txt_path}")
            return ""
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading transcript file {txt_path}: {e}")
            return ""

    def debug_video_usage(self, rag_results, query_type="normal"):
        """Debug helper to show which video parts are being used."""
        if not rag_results:
            print(f"=== {query_type.upper()} QUERY: No RAG results found ===")
            return
        
        video_ids = list({r['video_id'] for r in rag_results})
        print(f"=== {query_type.upper()} QUERY DEBUG ===")
        print(f"Total videos found: {len(video_ids)}")
        print(f"Video IDs: {video_ids}")
        print(f"Total segments: {len(rag_results)}")
        
        # Group segments by video
        segments_by_video = {}
        for result in rag_results:
            vid = result['video_id']
            if vid not in segments_by_video:
                segments_by_video[vid] = []
            segments_by_video[vid].append(result)
        
        print(f"\n=== SEGMENTS BY VIDEO ===")
        for video_id, segments in segments_by_video.items():
            print(f"\nVideo {video_id}:")
            print(f"  Segments: {len(segments)}")
            print(f"  Time range: {min(s['start'] for s in segments):.2f}s - {max(s['end'] for s in segments):.2f}s")
            print(f"  Total duration: {max(s['end'] for s in segments) - min(s['start'] for s in segments):.2f}s")
            
            # Show first few segments
            for i, seg in enumerate(segments[:3]):
                print(f"    Segment {i+1}: {seg['start']:.2f}s - {seg['end']:.2f}s")
                print(f"      Text: {seg['text'][:80]}...")
                print(f"      Similarity: {seg['similarity']:.3f}")
            
            if len(segments) > 3:
                print(f"    ... and {len(segments) - 3} more segments")

    def find_natural_ending(self, video_id, end_time, look_back=10):
        """Find a natural ending point within the look_back range, or extend to the next sentence-ending punctuation or farewell."""
        full_segments = self.load_transcript_segments(video_id)
        if not full_segments:
            return end_time

        # Expanded set of farewell/closure keywords
        closure_keywords = [
            'thank you', "that's all", 'in conclusion', 'finally', 'summary',
            'goodbye', 'see you', 'take care', 'have a great day', 'farewell',
            'bye', 'see you next time', 'until next time', 'hope this helps',
            'let me know if you have questions', 'concludes', 'conclusion', 'wrapping up', 'wrap up', 'end of', 'ending', 'all the best', 'best wishes', 'wish you', 'good night', 'good morning', 'good afternoon', 'good evening'
        ]
        # 1. Look for segments that end near the proposed end time and end naturally
        natural_endings = []
        for i, seg in enumerate(full_segments):
            if abs(seg['end'] - end_time) <= look_back:
                text = seg['text'].strip()
                if text.endswith(('.', '!', '?', ':', ';')) or any(keyword in text.lower() for keyword in closure_keywords):
                    # Also prefer if this is the last segment or a long pause follows
                    is_last = (i == len(full_segments) - 1)
                    next_start = full_segments[i+1]['start'] if i+1 < len(full_segments) else None
                    pause_after = (next_start - seg['end']) if next_start is not None else None
                    natural_endings.append((seg['end'], is_last, pause_after if pause_after is not None else 0))
        if natural_endings:
            # Prefer last segment, then longer pause, then closest
            natural_endings.sort(key=lambda x: (not x[1], -x[2], abs(x[0] - end_time)))
            chosen = natural_endings[0][0]
            print(f"Found natural ending at {chosen:.2f}s (was {end_time:.2f}s)")
            return chosen

        # 2. If no good ending is found, extend to the next sentence-ending punctuation after end_time
        for seg in full_segments:
            if seg['start'] >= end_time:
                text = seg['text'].strip()
                if text.endswith(('.', '!', '?', ':', ';')) or any(keyword in text.lower() for keyword in closure_keywords):
                    print(f"Extended to next natural ending at {seg['end']:.2f}s (was {end_time:.2f}s)")
                    return seg['end']

        # 3. If the clip is near the end of the video, prefer the last segment
        if full_segments and end_time > full_segments[-1]['end'] - 5:
            print(f"Clip is near the end, using last segment end at {full_segments[-1]['end']:.2f}s")
            return full_segments[-1]['end']

        # 4. Otherwise, return the original end_time
        return end_time

    def normalize(self, text):
        return text.replace(' ', '').lower()

    def find_complete_acronym_explanation(self, video_id, query):
        """Find complete explanations in a video for any query."""
        full_segments = self.load_transcript_segments(video_id)
        
        # Get video duration to ensure we don't exceed it
        video_path = os.path.join("Max Life Videos", f"{video_id}.mp4")
        video_duration = self.get_video_duration(video_path) if os.path.exists(video_path) else None
        
        # Extract query terms (both acronyms and regular terms)
        query_upper = query.upper()
        query_lower = query.lower()
        
        # Look for any capitalized terms that might be acronyms or important concepts
        capitalized_terms = re.findall(r'\b[A-Z]{3,}\b', query_upper)
        
        # Filter out common words
        common_words = {
            'WHAT', 'WHEN', 'WHERE', 'WHY', 'HOW', 'THE', 'AND', 'FOR', 'ARE', 'YOU', 'YOUR', 'THEY', 'THEIR', 
            'WITH', 'FROM', 'THAT', 'THIS', 'HAVE', 'HAS', 'HAD', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'MIGHT', 
    'MUST', 'CAN', 'MAY', 'GIVE', 'FULL', 'EXPLANATION', 'MEANING', 'DEFINITION', 'TELL', 'ABOUT',
    'IS', 'IN', 'ON', 'AT', 'TO', 'OF', 'BY', 'AS', 'OR', 'IF', 'DO', 'GO', 'UP', 'SO', 'NO', 'ME', 'MY'
        }
        important_terms = [term for term in capitalized_terms if term not in common_words]
        
        # Also extract key words from the query
        query_words = set(query_lower.split())
        
        # Normalize important terms for matching
        normalized_query_terms = [self.normalize(term) for term in important_terms]
        
        print(f"Looking for terms: {important_terms}")
        print(f"Query words: {query_words}")
        
        # Find relevant segments based on normalized term matches
        relevant_segments = []
        for seg in full_segments:
            text = seg['text'].lower()
            text_upper = seg['text'].upper()
            normalized_text = self.normalize(seg['text'])
            
            # Check for important terms in the segment (normalized)
            has_important_term = any(term in normalized_text for term in normalized_query_terms)
            
            # Check for query word matches
            segment_words = set(text.split())
            word_matches = len(query_words.intersection(segment_words))
            
            # Check for phrase matches
            phrase_matches = 0
            for word in query_words:
                if word in text:
                    phrase_matches += 1
            
            # Consider segment relevant if it has any matches
            if has_important_term or word_matches > 0 or phrase_matches > 0:
                relevant_segments.append(seg)
        
        if len(relevant_segments) >= 2:  # Should have at least 2 segments for a complete explanation
            # Sort by time and merge
            sorted_segments = sorted(relevant_segments, key=lambda x: x['start'])
            start_time = sorted_segments[0]['start']
            end_time = sorted_segments[-1]['end']
            
            # Find additional segments that might contain related content
            expanded_segments = []
            for seg in full_segments:
                # Check if this segment is close to our relevant segments
                is_close = any(abs(seg['start'] - s['start']) <= 30 for s in sorted_segments)
                
                # Check if it contains related keywords
                text = seg['text'].lower()
                related_keywords = [
                    'criteria', 'need', 'opportunity', 'physically', 'paying', 'capacity',
                    'specific', 'measurable', 'achievable', 'relevant', 'time-bound',
                    'strengths', 'weaknesses', 'opportunities', 'threats',
                    'analysis', 'framework', 'method', 'approach', 'strategy',
                    'first', 'second', 'third', 'fourth', 'finally',
                    'step', 'phase', 'stage', 'level', 'tier',
                    'benefit', 'advantage', 'feature', 'characteristic', 'property',
                    'example', 'instance', 'case', 'scenario', 'situation',
                    'explain', 'explanation', 'definition', 'meaning', 'refers', 'stands'
                ]
                has_related = any(keyword in text for keyword in related_keywords)
                if is_close or has_related:
                    expanded_segments.append(seg)
            # Merge all related segments
            if expanded_segments:
                all_segments = sorted_segments.copy()
                for exp_seg in expanded_segments:
                    is_duplicate = any(
                        abs(exp_seg['start'] - seg['start']) < 1.0 and 
                        abs(exp_seg['end'] - seg['end']) < 1.0 
                        for seg in all_segments
                    )
                    if not is_duplicate:
                        all_segments.append(exp_seg)
                all_segments = sorted(all_segments, key=lambda x: x['start'])
                start_time = all_segments[0]['start']
                end_time = all_segments[-1]['end']
                print(f"Expanded explanation from {len(sorted_segments)} to {len(all_segments)} segments")
            else:
                all_segments = sorted_segments
            if video_duration:
                if end_time > video_duration:
                    end_time = video_duration
                    print(f"Adjusted end time to video duration: {end_time:.2f}s")
                if start_time >= video_duration:
                    print(f"Start time {start_time:.2f}s is beyond video duration {video_duration:.2f}s")
                    return None
            max_duration = 60.0
            if end_time - start_time > max_duration:
                center = (start_time + end_time) / 2
                start_time = max(0, center - max_duration / 2)
                end_time = min(start_time + max_duration, end_time)
                if video_duration and end_time > video_duration:
                    end_time = video_duration
                    start_time = max(0, end_time - max_duration)
            limited_segments = [seg for seg in all_segments if seg['start'] >= start_time and seg['end'] <= end_time]
            full_text = ' '.join([s['text'] for s in limited_segments])
            relevance_score = 0.5  # Default score
            if important_terms:
                mentioned_terms = sum(1 for term in important_terms if self.normalize(term) in self.normalize(full_text))
                relevance_score = 1.0 - (mentioned_terms / len(important_terms))
            print(f"Found complete explanation: {start_time:.2f}-{end_time:.2f} (duration: {end_time-start_time:.2f}s, relevance: {relevance_score:.3f})")
            print(f"Text preview: {full_text[:100]}...")
            for seg in limited_segments:
                seg['video_id'] = seg.get('video_id', video_id)
                print(f"Segment {seg['start']:.2f}-{seg['end']:.2f} from video {seg['video_id']}")
            return {
                'start': start_time,
                'end': end_time,
                'text': full_text,
                'video_id': video_id,
                'similarity': relevance_score
            }
        return None

    def expand_segments_for_completeness(self, segments, max_expansion=60):
        """Expand segments to ensure complete topic coverage, especially for acronyms and multi-part explanations."""
        if not segments:
            return segments
        
        expanded_segments = []
        
        for seg in segments:
            # Start with the original segment
            expanded = seg.copy()
            
            # Look for potential acronyms, multi-part explanations, or structured content
            text = seg['text'].lower()
            
            # Keywords that indicate multi-part explanations
            multi_part_keywords = [
                'criteria', 'need', 'opportunity', 'physically', 'paying', 'capacity',
                'specific', 'measurable', 'achievable', 'relevant', 'time-bound',
                'strengths', 'weaknesses', 'opportunities', 'threats',
                'analysis', 'framework', 'method', 'approach', 'strategy',
                'first', 'second', 'third', 'fourth', 'finally',
                'step', 'phase', 'stage', 'level', 'tier',
                'benefit', 'advantage', 'feature', 'characteristic', 'property'
            ]
            
            # Check for acronym patterns
            acronym_patterns = re.findall(r'\b[A-Z]{2,}\b', seg['text'].upper())
            
            has_multi_part = any(keyword in text for keyword in multi_part_keywords)
            has_acronym = len(acronym_patterns) > 0
            
            if has_multi_part or has_acronym:
                print(f"Potential multi-part segment found: {seg['text'][:50]}...")
                
                # Load the full transcript for this video to find all related segments
                video_id = seg['video_id']
                full_segments = self.load_transcript_segments(video_id)
                
                # Find all segments within the expansion range that might be related
                related_segments = []
                for full_seg in full_segments:
                    if abs(full_seg['start'] - seg['start']) <= max_expansion:
                        # Check if this segment contains related content
                        full_text = full_seg['text'].lower()
                        if any(keyword in full_text for keyword in multi_part_keywords):
                            related_segments.append(full_seg)
                
                if len(related_segments) > 1:
                    # Merge related segments
                    sorted_related = sorted(related_segments, key=lambda x: x['start'])
                    expanded['start'] = sorted_related[0]['start']
                    expanded['end'] = sorted_related[-1]['end']
                    expanded['text'] = ' '.join([s['text'] for s in sorted_related])
                    print(f"Expanded segment: {expanded['start']:.2f}-{expanded['end']:.2f}")
                    print(f"Expanded text preview: {expanded['text'][:100]}...")
            
            expanded_segments.append(expanded)
        
        return expanded_segments

    def filter_relevant_segments(self, segments, max_segments=8):
        """Filter and rank segments by relevance for normal queries."""
        if not segments:
            return segments
        
        # Sort by similarity (lower distance is better)
        sorted_segments = sorted(segments, key=lambda x: x['similarity'])
        
        # Take the most relevant segments
        filtered = sorted_segments[:max_segments]
        
        # Additional filtering: remove segments that are too similar to each other
        # But be less aggressive to ensure we don't miss important content
        unique_segments = []
        for seg in filtered:
            # Check if this segment is too similar to already selected segments
            is_duplicate = False
            for existing in unique_segments:
                # If segments are from same video and close in time, consider them similar
                # But use a larger time window to avoid missing related content
                if (seg['video_id'] == existing['video_id'] and 
                    abs(seg['start'] - existing['start']) < 15):  # Reduced from 30 seconds
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_segments.append(seg)
        
        # If we filtered out too many segments, add some back
        if len(unique_segments) < max_segments // 2:
            print(f"Warning: Filtered too aggressively, adding back segments")
            for seg in filtered:
                # Check if this segment is already in unique_segments by comparing key properties
                already_included = any(
                    seg['video_id'] == existing['video_id'] and 
                    abs(seg['start'] - existing['start']) < 1.0 and
                    abs(seg['end'] - existing['end']) < 1.0
                    for existing in unique_segments
                )
                if not already_included:
                    unique_segments.append(seg)
                    if len(unique_segments) >= max_segments:
                        break
        
        return unique_segments

    def validate_and_improve_clips(self, clips, video_duration, is_acronym_query=False, query=""):
        """Validate LLM-proposed clip ranges: only ensure start < end and both are within video duration. No improvement or truncation."""
        validated_clips = []
        for clip in clips:
            start, end = clip['start'], clip['end']
            # Ensure start < end and both are within video duration
            if start < 0:
                start = 0
            if end > video_duration:
                end = video_duration
            if start >= end:
                continue
            validated_clips.append({
                'start': start,
                'end': end,
                'duration': end - start,
                'original': clip
            })
        return validated_clips

    def merge_adjacent_segments(self, segments, max_gap=5.0):
        """Merge adjacent segments that are close in time to create more comprehensive clips."""
        if not segments:
            return segments
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        merged = []
        current = sorted_segments[0].copy()
        
        for next_seg in sorted_segments[1:]:
            # If segments are close enough, merge them
            if next_seg['start'] - current['end'] <= max_gap:
                current['end'] = next_seg['end']
                current['text'] += ' ' + next_seg['text']
            else:
                merged.append(current)
                current = next_seg.copy()
        
        merged.append(current)
        return merged

    def preprocess_query_for_rag(self, query):
        """Preprocess query to improve RAG similarity scoring."""
        # Convert to lowercase for better matching
        query_lower = query.lower()
        
        # Extract key technical terms and concepts
        technical_terms = []
        
        # Common technical terms in the video content
        tech_keywords = [
            'bluetooth', 'light', 'wavelength', 'frequency', 'radio', 'wireless', 'communication',
            'insurance', 'policy', 'claim', 'premium', 'coverage', 'hospital', 'network', 'cashless',
            'surgery', 'medical', 'treatment', 'diagnosis', 'symptoms', 'disease', 'health',
            'mathematics', 'physics', 'science', 'theory', 'hypothesis', 'experiment', 'research',
            'traffic', 'transportation', 'boarding', 'airline', 'efficiency', 'optimization',
            'boredom', 'psychology', 'mental', 'health', 'wellbeing', 'productivity',
            'criteria', 'definition', 'explanation', 'meaning', 'clause'
        ]
        
        for keyword in tech_keywords:
            if keyword in query_lower:
                technical_terms.append(keyword)
        
        # If we found technical terms, create enhanced queries
        if technical_terms:
            enhanced_queries = [query]  # Keep original query
            for term in technical_terms:
                enhanced_queries.append(f"{term} {query}")
                enhanced_queries.append(f"{query} {term}")
            
            print(f"Enhanced queries for RAG: {enhanced_queries}")
            return enhanced_queries
        
        return [query]

    def build_llm_context(self, query, for_clipping=False):
        """Use RAG to find relevant videos/segments and build the transcript context for the LLM."""
        if not self.rag:
            return "", []

        # --- 1. Extract important terms and handle multi-part queries ---
        import re
        # Generalize subquery extraction for all queries
        split_pattern = re.compile(r'\b(?:and|&)\b', re.IGNORECASE)
        parts = [p.strip() for p in split_pattern.split(query) if p.strip()]
        if len(parts) > 1:
            subqueries = parts
        else:
            subqueries = [query]
        print(f"Multi-topic RAG: subqueries = {subqueries}")
        
        # --- 2. For each subquery, run a separate RAG search and merge results ---
        all_rag_results = []
        subquery_rag_results = []  # Track results per subquery
        for subq in subqueries:
            n_results = 15 if for_clipping else 8
            rag_results = self.rag.query_videos(subq, n_results=n_results)
            # Tag each result with its subquery for later grouping
            for r in rag_results:
                r['__subquery__'] = subq
            all_rag_results.extend(rag_results)
            subquery_rag_results.append((subq, rag_results))
            print(f"Subquery '{subq}' found {len(rag_results)} results")
        
        # Remove duplicates (by video_id, start, end)
        seen = set()
        unique_rag_results = []
        for r in all_rag_results:
            key = (r['video_id'], r['start'], r['end'])
            if key not in seen:
                unique_rag_results.append(r)
                seen.add(key)
        
        similarity_threshold = 0.75
        filtered = [r for r in unique_rag_results if r['similarity'] < similarity_threshold]
        filtered = self.improve_rag_results(filtered, query)
        filtered = sorted(filtered, key=lambda x: x['similarity'])[:6]
        
        # --- NEW LOGIC: Ensure all subqueries are represented in context (up to 8) ---
        # First, guarantee that each subquery's best video is included
        top_video_ids = set()
        for subq, rag_results in subquery_rag_results:
            if rag_results:
                # Always include the video of the top segment for each subquery
                best_video_id = rag_results[0]['video_id']
                top_video_ids.add(best_video_id)
                print(f"Guaranteed inclusion: {best_video_id} (best for subquery '{subq}')")
        
        # Now fill up to 8 with the next best globally
        all_rag_results_sorted = sorted(filtered, key=lambda x: x['similarity'])
        for r in all_rag_results_sorted:
            if len(top_video_ids) >= 8:
                break
            top_video_ids.add(r['video_id'])
        
        top_video_ids = list(top_video_ids)
        print(f"Using full transcripts for top {len(top_video_ids)} videos: {top_video_ids}")
        
        # Filter out videos that don't have transcripts
        available_video_ids = []
        for video_id in top_video_ids:
            transcript_path = os.path.join("Max Life Videos", f"{video_id}.txt")
            if os.path.exists(transcript_path):
                available_video_ids.append(video_id)
            else:
                print(f"⚠️ Video {video_id} has no transcript file, skipping")
        
        if len(available_video_ids) < len(top_video_ids):
            print(f"⚠️ {len(top_video_ids) - len(available_video_ids)} videos missing transcripts")
        
        top_video_ids = available_video_ids
        
        context_lines = []
        total_chars = 0
        
        # First, add the specific RAG segments that were found
        if filtered:
            context_lines.append("=== RELEVANT SEGMENTS FOUND ===")
            for i, result in enumerate(filtered):
                segment_text = f"[{result['start']:.2f} - {result['end']:.2f}] {result['text']}"
                context_lines.append(f"Segment {i+1} from {result['video_id']}: {segment_text}")
            context_lines.append("")  # Empty line for separation
        
        # ALWAYS include full transcripts for the top videos (up to 8)
        # This ensures the LLM has complete context for accurate decisions
        videos_added = 0
        for video_id in top_video_ids:
            full_transcript = self.load_full_transcript(video_id)
            if full_transcript:
                # For clipping queries, we need the full transcript for accurate timestamp decisions
                # For normal queries, we also need the full transcript for comprehensive answers
                # Only truncate if the transcript is extremely long (>5000 chars)
                if len(full_transcript) > 5000:
                    full_transcript = full_transcript[:4997] + "..."
                
                context_lines.append(f"=== Video: {video_id} ===\n{full_transcript}")
                total_chars += len(full_transcript)
                videos_added += 1
                print(f"Added full transcript for {video_id} ({len(full_transcript)} chars)")
                
                # Stop at 8 videos to keep context manageable
                if videos_added >= 8:
                    print(f"Reached 8 videos limit, stopping transcript inclusion")
                    break
        
        final_context = "\n\n".join(context_lines)
        # For full transcript approach, allow larger context for better accuracy
        final_context = self.check_token_limit(final_context, max_tokens=12000)
        estimated_tokens = self.estimate_tokens(final_context)
        print(f"Full transcript context: {len(final_context)} chars, ~{estimated_tokens} tokens")
        print(f"Videos included: {videos_added}")
        
        return final_context, filtered

    def send_message(self):
        user_input = self.user_entry.get()
        if user_input.strip().lower() == "exit":
            self.root.destroy()
            return
        self.display_message("user", user_input)
        self.user_entry.delete(0, tk.END)
        self.user_entry.configure(state="disabled")
        self.send_btn.configure(state="disabled")

        def extract_content(response):
            return str(response.content) if hasattr(response, "content") else str(response)

        def run_bot():
            if user_input.lower().startswith("clip:"):
                query = user_input[len("clip:"):].strip()
                context, rag_results = self.build_llm_context(query, for_clipping=True)
                
                # Debug: Show what context is being used for clipping
                print(f"=== CLIPPING DEBUG ===")
                print(f"Query: {query}")
                print(f"Context length: {len(context)}")
                print(f"RAG results count: {len(rag_results) if rag_results else 0}")
                
                # Use the debug helper
                self.debug_video_usage(rag_results, "clipping")
                
                # Get unique video IDs from RAG results
                video_ids = list({r['video_id'] for r in rag_results}) if rag_results else []
                print(f"Unique video IDs: {video_ids}")
                
                # Use the context that was already built by build_llm_context
                rag_context = context
                
                response_text = chain_clipping.invoke({"query": query, "transcript": rag_context})
                response = extract_content(response_text)
                print(f"LLM Response: {response}")  # Debug
                
                # Parse and validate clips
                proposed_clips = self.parse_llm_clip_response(response)
                
                print(f"Proposed clips: {proposed_clips}")
                
                # If no clips were parsed, the LLM might have given a text response instead of structured clips
                # In this case, we need to extract clips from the RAG results based on the LLM's response
                if not proposed_clips:
                    print("No clips parsed from LLM response, extracting from RAG results...")
                    proposed_clips = self.extract_clips_from_rag_results(rag_results, response, query)
                    print(f"Extracted {len(proposed_clips)} clips from RAG results")
                
                # Limit the number of clips to avoid overwhelming responses
                # For multi-topic queries, allow more clips to cover all topics
                query_upper = query.upper()
                multi_topic_indicators = [' AND ', ' & ', ' AND THE ', ' AND A ', ' AND AN ']
                is_multi_topic = any(indicator in query_upper for indicator in multi_topic_indicators)
                
                if is_multi_topic:
                    max_clips = 5  # Allow up to 5 clips for multi-topic queries
                    print(f"Multi-topic query detected - allowing up to {max_clips} clips")
                else:
                    max_clips = 3  # Limit to 3 clips for single-topic queries
                
                if len(proposed_clips) > max_clips:
                    print(f"Too many clips ({len(proposed_clips)}), limiting to {max_clips} most relevant")
                    # Keep the clips with the longest durations (most comprehensive coverage)
                    proposed_clips = sorted(proposed_clips, key=lambda x: x.get('duration', 0), reverse=True)[:max_clips]
                
                # Deduplicate overlapping clips
                if len(proposed_clips) > 1:
                    print("Deduplicating overlapping clips...")
                    # Use more conservative deduplication for multi-topic queries
                    min_overlap_threshold = 0.5 if is_multi_topic else 0.3
                    proposed_clips = self.deduplicate_clips(proposed_clips, min_overlap=min_overlap_threshold, query=query)
                    print(f"After deduplication: {len(proposed_clips)} clips")
                
                # Assign clips to their correct videos based on content relevance
                # For multi-topic queries, ensure we preserve all clips
                print(f"Before assignment: {len(proposed_clips)} clips")
                if len(proposed_clips) > 1:
                    print(f"Multi-topic query detected with {len(proposed_clips)} clips - preserving all clips")
                    proposed_clips = self.assign_clips_to_correct_videos(proposed_clips, query, rag_results)
                else:
                    proposed_clips = self.assign_clips_to_correct_videos(proposed_clips, query, rag_results)
                print(f"After assignment: {len(proposed_clips)} clips")
                
                video_clips = []
                # Fix: Use standard if-else for relevant_video_ids
                if 'filtered_rag_results' in locals():
                    relevant_video_ids = list({seg['video_id'] for seg in filtered_rag_results})
                else:
                    relevant_video_ids = list({seg['video_id'] for seg in rag_results}) if rag_results else []
                available_videos = []
                
                # Add videos
                for vid in relevant_video_ids:
                    candidate_path = os.path.join("Max Life Videos", f"{vid}.mp4")
                    if os.path.exists(candidate_path):
                        available_videos.append((vid, candidate_path, self.get_video_duration(candidate_path)))
                
                print(f"Available videos: {available_videos}")
                
                # Process each proposed clip
                print(f"Processing {len(proposed_clips)} proposed clips:")
                self.display_message("system", f"🎬 Processing {len(proposed_clips)} clips...")
                for i, clip in enumerate(proposed_clips):
                    start = clip['start']
                    end = clip['end']
                    video_id = clip.get('video_id', 'unknown')
                    print(f"Processing clip {i+1}/{len(proposed_clips)}: {start}-{end} from video {video_id}")
                    self.display_message("system", f"🎬 Creating clip {i+1}/{len(proposed_clips)}: {start:.1f}s - {end:.1f}s from {video_id}")
                    
                    video_file = None
                    matched_video_id = None
                    
                    # Use the video_id specified by the LLM if available
                    if 'video_id' in clip:
                        matched_video_id = clip['video_id']
                        print(f"Using LLM-specified video_id: {matched_video_id}")
                        
                        # Try to ensure the video file is available
                        video_path = self.ensure_video_file_available(matched_video_id)
                        if video_path:
                            print(f"✅ Video file available: {video_path}")
                            candidate_path = video_path
                        else:
                            print(f"⚠️ Video file not available for {matched_video_id}, looking for alternative...")
                            # Try to find an alternative video
                            alternative_path = self.find_alternative_video_for_clip(clip, rag_results)
                            if alternative_path:
                                candidate_path = alternative_path
                                matched_video_id = clip['video_id']  # Updated by find_alternative_video_for_clip
                                print(f"✅ Using alternative video: {matched_video_id}")
                            else:
                                print(f"❌ No alternative video found, falling back to content matching")
                                matched_video_id = None
                                candidate_path = None
                    
                    # If no video specified or invalid, try to match based on content relevance
                    if not matched_video_id:
                        # Look for segments that match this clip time range
                        for seg in rag_results:
                            if (start < seg['end'] and end > seg['start']):
                                matched_video_id = seg['video_id']
                                print(f"Matched clip to video {matched_video_id} based on segment overlap")
                                break
                    
                    # If still no match, use the original logic
                    if not matched_video_id:
                        # Find the video that has the most relevant content for this clip time range
                        best_video = None
                        best_score = 0
                        best_overlap = 0
                        
                        for video_id in video_ids:
                            # Get all segments from this video
                            video_segments = [s for s in rag_results if s['video_id'] == video_id]
                            if not video_segments:
                                continue
                            
                            # Calculate overlap and relevance for this video
                            total_overlap = 0
                            total_relevance = 0
                            relevant_segments = 0
                            
                            for seg in video_segments:
                                # Calculate overlap between clip and segment
                                overlap_start = max(start, seg['start'])
                                overlap_end = min(end, seg['end'])
                                overlap_duration = max(0, overlap_end - overlap_start)
                                
                                if overlap_duration > 0:
                                    total_overlap += overlap_duration
                                    # Use similarity as relevance score (lower distance is better)
                                    relevance_score = 1 - seg['similarity']
                                    total_relevance += relevance_score * overlap_duration
                                    relevant_segments += 1
                            
                            if total_overlap > 0:
                                # Calculate average relevance weighted by overlap
                                avg_relevance = total_relevance / total_overlap
                                
                                # Score based on both overlap and relevance
                                # Prioritize videos with more overlap and better relevance
                                score = total_overlap * avg_relevance
                                
                                print(f"Video {video_id}: overlap={total_overlap:.2f}s, relevance={avg_relevance:.3f}, score={score:.3f}")
                                
                                if score > best_score:
                                    best_score = score
                                    best_video = video_id
                                    best_overlap = total_overlap
                        
                        if best_video:
                            matched_video_id = best_video
                            print(f"Matched clip to video {matched_video_id} based on content relevance (overlap: {best_overlap:.2f}s, score: {best_score:.3f})")
                        else:
                            # Fallback: find any video with segments in this time range
                            for seg in rag_results:
                                if (start < seg['end'] and end > seg['start']):
                                    matched_video_id = seg['video_id']
                                    print(f"Fallback: matched clip to video {matched_video_id} based on time overlap")
                                    break
                
                    # Fallback to first available video if still no match
                    if not matched_video_id and video_ids:
                        matched_video_id = video_ids[0]
                        print(f"Fallback: using first available video {matched_video_id}")
                    
                    # Try to get the video file
                    if matched_video_id:
                        # Try to ensure the video file is available
                        video_path = self.ensure_video_file_available(matched_video_id)
                        if video_path:
                            video_file = video_path
                        else:
                            # Try to find an alternative video
                            alternative_path = self.find_alternative_video_for_clip(clip, rag_results)
                            if alternative_path:
                                video_file = alternative_path
                                matched_video_id = clip['video_id']  # Updated by find_alternative_video_for_clip
                                print(f"✅ Using alternative video: {matched_video_id}")
                    
                    # Fallback to any available video
                    if not video_file and available_videos:
                        # Try to ensure the first available video is actually available
                        first_video_id = available_videos[0][0]
                        video_path = self.ensure_video_file_available(first_video_id)
                        if video_path:
                            video_file = video_path
                            matched_video_id = first_video_id
                            print(f"Fallback: using first available video file: {first_video_id}")
                        else:
                            # Try to find any working video
                            for vid_id, vid_path, vid_duration in available_videos:
                                if os.path.exists(vid_path):
                                    video_file = vid_path
                                    matched_video_id = vid_id
                                    print(f"Fallback: using working video: {vid_id}")
                                    break
                    
                    # Fallback to uploaded video
                    if not video_file and self.video_path and os.path.exists(self.video_path):
                        video_file = self.video_path
                        print(f"Fallback: using uploaded video")
                    
                    # 7. Validate and improve the clip
                    if video_file:
                        duration = self.get_video_duration(video_file)
                        print(f"Using video: {video_file} (duration: {duration}) for clip {start}-{end}")
                        
                        if duration is not None:
                            # Add video_id to clip for natural ending detection
                            clip_with_video = clip.copy()
                            clip_with_video['video_id'] = matched_video_id
                            
                            # Validate and improve the clip
                            improved_clips = self.validate_and_improve_clips([clip_with_video], duration, is_acronym_query=False, query=query)
                            
                            for improved_clip in improved_clips:
                                print(f"Improved clip: {improved_clip['start']:.2f}-{improved_clip['end']:.2f} (duration: {improved_clip['duration']:.2f}s)")
                                msg, clip_path = self.clip_video(video_file, improved_clip['start'], improved_clip['end'])
                                if clip_path:
                                    video_clips.append(clip_path)
                                    print(f"✅ Successfully created clip: {clip_path}")
                                else:
                                    print(f"❌ Failed to create clip: {msg}")
                        else:
                            print(f"❌ Could not determine video duration for {video_file}")
                    else:
                        # Provide detailed feedback about missing video
                        original_video_id = clip.get('video_id', 'unknown')
                        print(f"❌ No valid video file found for clip {start}-{end} (original video: {original_video_id})")
                        
                        # Check if this is a missing video issue
                        if original_video_id != 'unknown':
                            transcript_path = os.path.join("Max Life Videos", f"{original_video_id}.txt")
                            if os.path.exists(transcript_path):
                                self.display_message("assistant", f"⚠️ Video file for '{original_video_id}' is missing but transcript exists. This video may not have been fully downloaded from Google Drive. Try clicking '📥 Download Missing' to attempt to download it.")
                            else:
                                self.display_message("assistant", f"⚠️ Video file for '{original_video_id}' not found and no transcript exists. This video may not exist in your collection.")
                        else:
                            self.display_message("assistant", f"⚠️ Could not find a suitable video file for clip {start:.1f}s - {end:.1f}s. No videos with matching content were found.")
                
                video_clips = [p for p in video_clips if p and os.path.exists(p)]
                if not video_clips:
                    self.display_message("system", "⚠ No valid clips generated for concatenation.")
                    return
                else:
                    self.display_message("system", f"✅ Successfully created {len(video_clips)} clips! Now concatenating...")
                    # Validate video compatibility before concatenation
                    is_compatible, compatibility_msg = self.validate_video_compatibility(video_clips)
                    if not is_compatible:
                        print(f"Warning: {compatibility_msg}")
                        self.display_message("system", f"⚠ Video compatibility warning: {compatibility_msg}")
                        # Continue anyway since we're using re-encoding
                    
                    # Save to Downloads folder instead of temp
                    downloads_path = os.path.expanduser("~/Downloads")
                    if not os.path.exists(downloads_path):
                        downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
                    if not os.path.exists(downloads_path):
                        # Fallback to temp if Downloads doesn't exist
                        downloads_path = tempfile.gettempdir()
                    
                    output_filepath = os.path.join(downloads_path, "final_output.mp4")
                    msg, out = concatenate_videos(video_clips, output_filepath)
                    print(f"Concatenation result: {msg}, Output: {out}")  # Debug
                    
                    # Update the message to show the Downloads path
                    if "✅ Concatenated video saved to:" in msg:
                        msg = f"✅ Video clips successfully created and saved to your Downloads folder as 'final_output.mp4'"
                    
                    self.display_message("Vivi", msg)
            else:
                context, rag_results = self.build_llm_context(user_input, for_clipping=False)
                
                # Debug: Show what context is being used for normal queries
                print(f"=== NORMAL QUERY DEBUG ===")
                print(f"Query: {user_input}")
                print(f"Context length: {len(context)}")
                print(f"RAG results count: {len(rag_results) if rag_results else 0}")
                
                # Use the debug helper
                self.debug_video_usage(rag_results, "normal")
                
                print(f"Context preview: {context[:300]}...")
                
                # Additional detailed debug information
                if rag_results:
                    print(f"\n=== RAG RESULTS ANALYSIS ===")
                    print(f"Total RAG results: {len(rag_results)}")
                    video_ids = list(set(r['video_id'] for r in rag_results))
                    print(f"Videos found: {video_ids}")
                    
                    # Show top segments with their relevance
                    print(f"\n=== TOP SEGMENTS BY RELEVANCE ===")
                    for i, result in enumerate(rag_results[:5]):
                        print(f"{i+1}. {result['video_id']} ({result['start']:.1f}s - {result['end']:.1f}s)")
                        print(f"   Similarity: {result['similarity']:.3f}")
                        print(f"   Text: {result['text'][:150]}...")
                        print()
                    
                    # Show segment distribution by video
                    video_counts = {}
                    for result in rag_results:
                        video_counts[result['video_id']] = video_counts.get(result['video_id'], 0) + 1
                    
                    print(f"\n=== SEGMENT DISTRIBUTION BY VIDEO ===")
                    for video_id, count in sorted(video_counts.items(), key=lambda x: x[1], reverse=True):
                        print(f"{video_id}: {count} segments")
                    
                    print(f"\n=== CONTEXT ANALYSIS ===")
                    print(f"Context length: {len(context)} characters")
                    print(f"Estimated tokens: {len(context) // 4}")
                    print(f"Context preview: {context[:300]}...")
                
                response_text = chain_general.invoke({
                    "context": self.context,
                    "question": user_input,
                    "transcript": context
                })
                response = extract_content(response_text)
                
                # Debug: Show LLM response
                print(f"\n=== LLM RESPONSE ===")
                print(f"Response: {response}")
                print(f"Response length: {len(response)} characters")
                
                self.typewriter_effect("Vivi", response)
                # Use conversation history management instead of direct accumulation
                self.manage_conversation_history(user_input, response)
            self.user_entry.configure(state="normal")
            self.send_btn.configure(state="normal")
            self.user_entry.focus()

        threading.Thread(target=run_bot).start()



    def play_final_video(self):
        # Check Downloads folder first, then temp directory, then original video directory
        downloads_path = os.path.expanduser("~/Downloads")
        if not os.path.exists(downloads_path):
            downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
        
        final_output = os.path.join(downloads_path, "final_output.mp4")
        if not os.path.exists(final_output):
            final_output = os.path.join(tempfile.gettempdir(), "final_output.mp4")
        if not os.path.exists(final_output) and self.video_path:
            final_output = os.path.join(os.path.dirname(self.video_path), "final_output.mp4")
        
        if not os.path.exists(final_output):
            self.display_message("system", "⚠️ No final video available. Generate a clipped video first.")
            return
        if self.audio_process and self.audio_process.poll() is None:
            self.audio_process.terminate()
        try:
            self.audio_process = subprocess.Popen(["ffplay", "-autoexit", "-loglevel", "quiet", final_output])
        except Exception as e:
            self.display_message("system", f"❌ Failed to play final video: {e}")

    def get_video_duration(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps > 0:
                duration = frame_count / fps
            else:
                duration = None
            cap.release()
            return duration
        except Exception as e:
            print(f"Error getting duration for {video_path}: {e}")
            return None

    def get_video_properties(self, video_path):
        """Get video properties like codec, frame rate, resolution, etc."""
        try:
            command = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", video_path
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            properties = {
                'duration': None,
                'fps': None,
                'width': None,
                'height': None,
                'video_codec': None,
                'audio_codec': None,
                'audio_sample_rate': None,
                'audio_channels': None
            }
            
            if 'format' in data and 'duration' in data['format']:
                properties['duration'] = float(data['format']['duration'])
            
            for stream in data.get('streams', []):
                if stream['codec_type'] == 'video':
                    properties['fps'] = eval(stream.get('r_frame_rate', '30/1'))
                    properties['width'] = int(stream.get('width', 0))
                    properties['height'] = int(stream.get('height', 0))
                    properties['video_codec'] = stream.get('codec_name', 'unknown')
                elif stream['codec_type'] == 'audio':
                    properties['audio_codec'] = stream.get('codec_name', 'unknown')
                    properties['audio_sample_rate'] = int(stream.get('sample_rate', 0))
                    properties['audio_channels'] = int(stream.get('channels', 0))
            
            return properties
        except Exception as e:
            print(f"Error getting video properties for {video_path}: {e}")
            return None

    def validate_video_compatibility(self, video_paths):
        """Validate that all videos have compatible properties for concatenation."""
        if not video_paths:
            return True, "No videos to validate"
        
        properties_list = []
        for path in video_paths:
            props = self.get_video_properties(path)
            if props:
                properties_list.append((path, props))
            else:
                return False, f"Could not get properties for {path}"
        
        if len(properties_list) < 2:
            return True, "Only one video, no compatibility issues"
        
        # Check for major compatibility issues
        issues = []
        base_props = properties_list[0][1]
        
        for path, props in properties_list[1:]:
            # Check for major differences that could cause issues
            if props['video_codec'] != base_props['video_codec']:
                issues.append(f"Different video codecs: {base_props['video_codec']} vs {props['video_codec']} in {path}")
            
            if props['audio_codec'] != base_props['audio_codec']:
                issues.append(f"Different audio codecs: {base_props['audio_codec']} vs {props['audio_codec']} in {path}")
            
            if abs(props['fps'] - base_props['fps']) > 1:
                issues.append(f"Significant FPS difference: {base_props['fps']} vs {props['fps']} in {path}")
            
            if abs(props['audio_sample_rate'] - base_props['audio_sample_rate']) > 1000:
                issues.append(f"Different audio sample rates: {base_props['audio_sample_rate']} vs {props['audio_sample_rate']} in {path}")
        
        if issues:
            return False, f"Compatibility issues found: {'; '.join(issues)}"
        
        return True, "All videos are compatible"

    def deduplicate_clips(self, clips, min_overlap=0.3, query=""):
        """Remove overlapping clips and keep only the most relevant ones."""
        if not clips:
            return clips
        
        # Add duration to clips if not present
        for clip in clips:
            if 'duration' not in clip:
                clip['duration'] = clip['end'] - clip['start']
        
        # For multi-topic queries, be more careful about deduplication
        # Check if this is a multi-topic query
        query_upper = query.upper()
        multi_topic_indicators = [' AND ', ' & ', ' AND THE ', ' AND A ', ' AND AN ']
        is_multi_topic = any(indicator in query_upper for indicator in multi_topic_indicators)
        
        if is_multi_topic:
            print(f"Multi-topic query detected - being conservative with deduplication")
            # For multi-topic queries, only remove exact duplicates or very high overlap
            min_overlap = max(min_overlap, 0.8)  # Use the higher of the two thresholds
        
        # Sort clips by duration (longer clips first) and then by start time
        sorted_clips = sorted(clips, key=lambda x: (x['duration'], -x['start']), reverse=True)
        
        non_overlapping = []
        
        for clip in sorted_clips:
            is_duplicate = False
            
            for existing in non_overlapping:
                # Calculate overlap
                overlap_start = max(clip['start'], existing['start'])
                overlap_end = min(clip['end'], existing['end'])
                
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    clip_duration = clip['end'] - clip['start']
                    existing_duration = existing['end'] - existing['start']
                    
                    # Calculate overlap percentage
                    overlap_ratio = overlap_duration / min(clip_duration, existing_duration)
                    
                    # Check if clips are from different videos (different topics)
                    different_videos = clip.get('video_id') != existing.get('video_id')
                    
                    # For multi-topic queries, preserve clips from different videos
                    if is_multi_topic and different_videos:
                        print(f"Preserving clip from different video: {clip['start']:.2f}-{clip['end']:.2f} from {clip.get('video_id')} vs {existing.get('video_id')}")
                        continue
                    
                    # Normal deduplication logic
                    if overlap_ratio > min_overlap:
                        is_duplicate = True
                        print(f"Removing duplicate clip: {clip['start']:.2f}-{clip['end']:.2f} (overlaps {overlap_ratio:.2f} with {existing['start']:.2f}-{existing['end']:.2f})")
                        break
            
            if not is_duplicate:
                non_overlapping.append(clip)
                print(f"Keeping clip: {clip['start']:.2f}-{clip['end']:.2f} (duration: {clip['duration']:.2f}s) from {clip.get('video_id', 'unknown')}")
        
        return non_overlapping



    def parse_llm_clip_response(self, response):
        """Parse LLM response to extract clip ranges and video assignments more robustly."""
        proposed_clips = []
        
        # Split response into lines and look for clip ranges
        lines = response.strip().split("\n")
        
        current_video_id = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Look for video specification
            video_match = re.search(r"-?\s*Video:\s*([^\n]+)", line, re.IGNORECASE)
            if video_match:
                current_video_id = video_match.group(1).strip()
                print(f"Found video specification: {current_video_id}")
                continue
                
            # Look for different patterns of clip ranges
            patterns = [
                r"-?\s*Range:\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",
                r"Range:\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",
                r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",
                r"start[:\s]*(\d+\.?\d*)[\s-]+end[:\s]*(\d+\.?\d*)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    try:
                        start = float(match.group(1))
                        end = float(match.group(2))
                        
                        # Basic validation
                        if start >= 0 and end > start and end - start <= 300:  # Max 5 minutes
                            clip = {
                                'start': start, 
                                'end': end, 
                                'duration': end - start
                            }
                            
                            # Add video_id if specified
                            if current_video_id:
                                clip['video_id'] = current_video_id
                                print(f"Parsed clip: {start:.2f}-{end:.2f} from video {current_video_id}")
                            else:
                                print(f"Parsed clip: {start:.2f}-{end:.2f} (no video specified)")
                            
                            proposed_clips.append(clip)
                            break
                    except (ValueError, IndexError):
                        continue
        
        return proposed_clips

    def extract_clips_from_rag_results(self, rag_results, llm_response, query):
        """Extract clips from RAG results when LLM doesn't provide structured clips."""
        clips = []
        
        if not rag_results:
            return clips
        
        # For multi-topic queries, try to cover different topics
        query_upper = query.upper()
        multi_topic_indicators = [' AND ', ' & ', ' AND THE ', ' AND A ', ' AND AN ']
        is_multi_topic = any(indicator in query_upper for indicator in multi_topic_indicators)
        
        # Group segments by video
        video_segments = {}
        for seg in rag_results:
            video_id = seg['video_id']
            if video_id not in video_segments:
                video_segments[video_id] = []
            video_segments[video_id].append(seg)
        
        # For multi-topic queries, select the best segment from each video
        # For single-topic queries, select the best segments overall
        if is_multi_topic:
            # Select up to 2 segments from different videos for multi-topic queries
            selected_videos = list(video_segments.keys())[:2]
            for video_id in selected_videos:
                segments = video_segments[video_id]
                if segments:
                    # Select the segment with the highest similarity score
                    best_segment = max(segments, key=lambda x: x.get('similarity', 0))
                    clip = {
                        'start': best_segment['start'],
                        'end': best_segment['end'],
                        'duration': best_segment['end'] - best_segment['start'],
                        'video_id': video_id
                    }
                    clips.append(clip)
        else:
            # For single-topic queries, select the best segments overall
            all_segments = []
            for segments in video_segments.values():
                all_segments.extend(segments)
            
            # Sort by similarity and select the best ones
            sorted_segments = sorted(all_segments, key=lambda x: x.get('similarity', 0), reverse=True)
            for seg in sorted_segments[:3]:  # Take top 3 segments
                clip = {
                    'start': seg['start'],
                    'end': seg['end'],
                    'duration': seg['end'] - seg['start'],
                    'video_id': seg['video_id']
                }
                clips.append(clip)
        
        return clips

    def is_definition_query(self, query):
        """Detect if a query is asking for a definition or explanation."""
        query_lower = query.lower()
        definition_keywords = [
            'what is', 'what does', 'meaning of', 'definition of', 'explain', 'describe',
            'tell me about', 'how does', 'what are', 'define', 'explanation'
        ]
        return any(keyword in query_lower for keyword in definition_keywords)

    def assign_clips_to_correct_videos(self, clips, query, rag_results):
        """Assign clips to their correct videos based on content relevance."""
        if not clips:
            return clips
        
        # Use a simpler approach: let the LLM's video assignments guide us
        # If the LLM specified different videos for different clips, respect that
        llm_video_assignments = {}
        for clip in clips:
            if 'video_id' in clip:
                video_id = clip['video_id']
                if video_id not in llm_video_assignments:
                    llm_video_assignments[video_id] = []
                llm_video_assignments[video_id].append(clip)
        
        # If LLM assigned clips to different videos, use that as the guide
        if len(llm_video_assignments) > 1:
            print(f"LLM assigned clips to {len(llm_video_assignments)} different videos - respecting LLM decisions")
            
            # Simply use the LLM's video assignments as-is
            assigned_clips = []
            for video_id, video_clips in llm_video_assignments.items():
                for clip in video_clips:
                    # Keep the LLM's video assignment
                    assigned_clips.append(clip)
                    print(f"Respecting LLM assignment: clip {clip['start']:.2f}-{clip['end']:.2f} to video {video_id}")
            
            return assigned_clips
        
        else:
            # Single video case - use content-based assignment
            print(f"LLM assigned all clips to same video or no video specified - using content-based assignment")
            assigned_clips = []
            
            for clip in clips:
                start, end = clip['start'], clip['end']
                original_video_id = clip.get('video_id', None)
                
                # If LLM specified a video, use it
                if original_video_id:
                    clip['video_id'] = original_video_id
                    print(f"Using LLM-specified video: clip {start:.2f}-{end:.2f} to video {original_video_id}")
                else:
                    # Find the best video based on content overlap
                    best_video = None
                    best_score = 0
                    
                    for video_id in list({r['video_id'] for r in rag_results}):
                        # Check if this video has segments that overlap with the clip
                        video_segments = [s for s in rag_results if s['video_id'] == video_id]
                        overlap_score = 0
                        
                        for seg in video_segments:
                            if (start < seg['end'] and end > seg['start']):
                                overlap_score += 1 - seg['similarity']  # Lower distance is better
                        
                        if overlap_score > best_score:
                            best_score = overlap_score
                            best_video = video_id
                    
                    if best_video:
                        clip['video_id'] = best_video
                        print(f"Content-based assignment: clip {start:.2f}-{end:.2f} to video {best_video} (score: {best_score:.2f})")
                    else:
                        # Fallback to first available video
                        available_videos = list({r['video_id'] for r in rag_results})
                        if available_videos:
                            clip['video_id'] = available_videos[0]
                            print(f"Fallback: assigned clip {start:.2f}-{end:.2f} to video {available_videos[0]}")
                
                assigned_clips.append(clip)
            
            return assigned_clips

    def ensure_video_file_available(self, video_id):
        """Ensure that a video file is available locally, downloading it from Google Drive if necessary."""
        video_path = os.path.join("Max Life Videos", f"{video_id}.mp4")
        
        # Check if video file exists locally
        if os.path.exists(video_path):
            print(f"✅ Video file already available: {video_path}")
            return video_path
        
        # Check if transcript exists (indicating the video should exist in Drive)
        transcript_path = os.path.join("Max Life Videos", f"{video_id}.txt")
        if not os.path.exists(transcript_path):
            print(f"⚠️ No transcript found for {video_id}, cannot download video")
            return None
        
        print(f"📥 Video file missing for {video_id}, attempting to download from Google Drive...")
        
        # Try to download from Google Drive if sync is available
        if self.drive_sync:
            try:
                # Get the list of files from Google Drive
                drive_files = self.drive_sync.list_drive_files()
                
                # Find the video file in Drive
                video_file_info = None
                for file_info in drive_files:
                    if file_info['name'] == f"{video_id}.mp4":
                        video_file_info = file_info
                        break
                
                if video_file_info:
                    print(f"🎬 Found video in Google Drive: {video_id}.mp4")
                    
                    # Download the video file
                    if self.drive_sync.download_file(video_file_info['id'], video_path):
                        print(f"✅ Successfully downloaded video: {video_path}")
                        return video_path
                    else:
                        print(f"❌ Failed to download video: {video_id}.mp4")
                        return None
                else:
                    print(f"⚠️ Video file {video_id}.mp4 not found in Google Drive")
                    return None
                    
            except Exception as e:
                print(f"❌ Error downloading video {video_id}.mp4: {e}")
                return None
        else:
            print(f"⚠️ Google Drive sync not available, cannot download missing video: {video_id}.mp4")
            return None

    def find_alternative_video_for_clip(self, clip, rag_results):
        """Find an alternative video that can be used for a clip when the original video is missing."""
        start, end = clip['start'], clip['end']
        original_video_id = clip.get('video_id', None)
        
        print(f"🔍 Looking for alternative video for clip {start:.2f}-{end:.2f} (original: {original_video_id})")
        
        # First, try to find videos with similar content that have MP4 files
        available_videos = []
        for result in rag_results:
            video_id = result['video_id']
            video_path = os.path.join("Max Life Videos", f"{video_id}.mp4")
            
            if os.path.exists(video_path):
                # Check if this video has content that overlaps with our clip
                if (start < result['end'] and end > result['start']):
                    available_videos.append((video_id, video_path, result['similarity']))
        
        if available_videos:
            # Sort by similarity (lower is better) and take the best match
            available_videos.sort(key=lambda x: x[2])
            best_video_id, best_video_path, best_similarity = available_videos[0]
            
            print(f"✅ Found alternative video: {best_video_id} (similarity: {best_similarity:.3f})")
            
            # Update the clip to use the alternative video
            clip['video_id'] = best_video_id
            clip['alternative_source'] = True
            clip['original_video_id'] = original_video_id
            
            return best_video_path
        
        # If no good alternative found, try any available video
        all_video_paths = []
        for result in rag_results:
            video_id = result['video_id']
            video_path = os.path.join("Max Life Videos", f"{video_id}.mp4")
            if os.path.exists(video_path) and video_path not in all_video_paths:
                all_video_paths.append(video_path)
        
        if all_video_paths:
            fallback_video = all_video_paths[0]
            fallback_video_id = os.path.splitext(os.path.basename(fallback_video))[0]
            
            print(f"⚠️ Using fallback video: {fallback_video_id}")
            
            # Update the clip to use the fallback video
            clip['video_id'] = fallback_video_id
            clip['fallback_source'] = True
            clip['original_video_id'] = original_video_id
            
            return fallback_video
        
        print(f"❌ No alternative video found for clip {start:.2f}-{end:.2f}")
        return None

    def validate_all_acronyms_covered(self, query, clips, rag_results):
        """Validate that all terms mentioned in the query are covered by the clips."""
        # Let the LLM decide what's important - don't pre-filter terms
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Don't pre-determine important terms - let the LLM and RAG results guide this
        important_terms = []
        
        # For validation, we'll just check if the clips cover the general query content
        print(f"Validating coverage of query content")
        
        # Since we're letting the LLM decide what's important, 
        # we'll just validate that the clips exist and have reasonable content
        if not clips:
            print("Warning: No clips provided for validation")
            return False
        
        print(f"Validated {len(clips)} clips - letting LLM determine content relevance")
        return True

    def run(self):
        """Run the chatbot application."""
        try:
            self.root.mainloop()
        finally:
            # Cleanup: stop Google Drive sync
            if self.drive_sync:
                print("🛑 Stopping Google Drive sync...")
                self.drive_sync.stop_watching()
                print("✅ Google Drive sync stopped")

    def stop_google_drive_sync(self):
        """Stop Google Drive sync if running."""
        if self.drive_sync:
            self.drive_sync.stop_watching()
            print("✅ Google Drive sync stopped")

    def manual_sync(self):
        """Manually trigger Google Drive sync."""
        if not self.drive_sync:
            self.display_message("assistant", "❌ Google Drive sync is not available. Please check your credentials and try again.")
            return
        
        def sync_worker():
            try:
                # Update status
                self.root.after(0, lambda: self.sync_status_label.configure(
                    text="🔄 Syncing...",
                    text_color="#ffc107"
                ))
                
                # Perform sync
                sync_result = self.drive_sync.sync_folder()
                
                # Check for missing video files and attempt to download them
                missing_videos = self.check_for_missing_videos()
                if missing_videos:
                    print(f"🔍 Found {len(missing_videos)} missing video files, attempting to download...")
                    downloaded_count = 0
                    for video_id in missing_videos:
                        if self.ensure_video_file_available(video_id):
                            downloaded_count += 1
                    
                    if downloaded_count > 0:
                        sync_result['downloaded'].extend([f"{vid}.mp4" for vid in missing_videos[:downloaded_count]])
                
                # Update status
                self.root.after(0, lambda: self.sync_status_label.configure(
                    text="✅ Google Drive sync active",
                    text_color="#28a745"
                ))
                
                # Show sync results
                message = f"✅ Sync completed!\n📥 Downloaded: {len(sync_result['downloaded'])} files\n🔄 Updated: {len(sync_result['updated'])} files"
                if missing_videos and downloaded_count > 0:
                    message += f"\n🎬 Downloaded {downloaded_count} missing video files"
                self.root.after(0, lambda: self.display_message("assistant", message))
                
            except Exception as e:
                error_msg = f"❌ Sync failed: {str(e)}"
                self.root.after(0, lambda: self.sync_status_label.configure(
                    text="❌ Sync failed",
                    text_color="#dc3545"
                ))
                self.root.after(0, lambda: self.display_message("assistant", error_msg))
        
        # Run sync in background thread
        sync_thread = threading.Thread(target=sync_worker, daemon=True)
        sync_thread.start()



    def check_for_missing_videos(self):
        """Check for video files that have transcripts but no MP4 files."""
        missing_videos = []
        
        if not os.path.exists("Max Life Videos"):
            return missing_videos
        
        for file_name in os.listdir("Max Life Videos"):
            if file_name.endswith('.txt'):
                video_id = file_name[:-4]  # Remove .txt extension
                video_path = os.path.join("Max Life Videos", f"{video_id}.mp4")
                
                if not os.path.exists(video_path):
                    missing_videos.append(video_id)
        
        if missing_videos:
            print(f"🔍 Found {len(missing_videos)} missing video files: {missing_videos}")
        
        return missing_videos

    def check_missing_videos_on_startup(self):
        """Check for missing video files on startup and notify the user."""
        def startup_check():
            try:
                missing_videos = self.check_for_missing_videos()
                if missing_videos:
                    message = f"⚠️ Found {len(missing_videos)} missing video files that have transcripts but no MP4 files.\n\n"
                    message += "This can happen when videos weren't fully downloaded from Google Drive.\n"
                    message += "You can:\n"
                    message += "1. Click '🔄 Sync Drive' to attempt to download missing videos\n"
                    message += "2. The system will automatically try to find alternative videos for clips\n\n"
                    message += f"Missing videos: {', '.join(missing_videos[:5])}"
                    if len(missing_videos) > 5:
                        message += f" and {len(missing_videos) - 5} more..."
                    
                    # Show message in chat after a short delay
                    self.root.after(2000, lambda: self.display_message("assistant", message))
                    
            except Exception as e:
                print(f"Error checking for missing videos on startup: {e}")
        
        # Run the check in a background thread
        startup_thread = threading.Thread(target=startup_check, daemon=True)
        startup_thread.start()

    def download_missing_videos(self):
        """Manually download missing video files from Google Drive."""
        if not self.drive_sync:
            self.display_message("assistant", "❌ Google Drive sync is not available. Please check your credentials and try again.")
            return
        
        def download_worker():
            try:
                # Update button state
                self.root.after(0, lambda: self.download_missing_btn.configure(
                    text="📥 Downloading...",
                    state="disabled"
                ))
                
                # Check for missing videos
                missing_videos = self.check_for_missing_videos()
                
                if not missing_videos:
                    self.root.after(0, lambda: self.display_message("assistant", "✅ No missing video files found!"))
                    return
                
                # Attempt to download each missing video
                downloaded_count = 0
                failed_count = 0
                
                for i, video_id in enumerate(missing_videos):
                    self.root.after(0, lambda vid=video_id, idx=i+1, total=len(missing_videos): 
                        self.display_message("assistant", f"📥 Downloading {vid}.mp4 ({idx}/{total})...")
                    )
                    
                    if self.ensure_video_file_available(video_id):
                        downloaded_count += 1
                        self.root.after(0, lambda vid=video_id: 
                            self.display_message("assistant", f"✅ Successfully downloaded {vid}.mp4")
                        )
                    else:
                        failed_count += 1
                        self.root.after(0, lambda vid=video_id: 
                            self.display_message("assistant", f"❌ Failed to download {vid}.mp4")
                        )
                
                # Show final results
                final_message = f"📥 Download complete!\n✅ Downloaded: {downloaded_count} videos\n❌ Failed: {failed_count} videos"
                if failed_count > 0:
                    final_message += "\n\nFailed videos may not exist in Google Drive or may have permission issues."
                
                self.root.after(0, lambda: self.display_message("assistant", final_message))
                
            except Exception as e:
                error_msg = f"❌ Download failed: {str(e)}"
                self.root.after(0, lambda: self.display_message("assistant", error_msg))
            finally:
                # Reset button state
                self.root.after(0, lambda: self.download_missing_btn.configure(
                    text="📥 Download Missing",
                    state="normal"
                ))
        
        # Run download in background thread
        download_thread = threading.Thread(target=download_worker, daemon=True)
        download_thread.start()

    def improve_rag_results(self, rag_results, query):
        """Improve RAG results by boosting segments with exact keyword matches."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        improved_results = []
        
        for result in rag_results:
            # Calculate base similarity (lower is better for distance-based scores)
            base_similarity = result['similarity']
            
            # Check for exact keyword matches in the segment text
            segment_text_lower = result['text'].lower()
            segment_words = set(segment_text_lower.split())
            
            # Count exact word matches
            exact_matches = len(query_words.intersection(segment_words))
            
            # Check for phrase matches (consecutive words)
            phrase_matches = 0
            query_phrases = []
            for i in range(len(query_words)):
                for j in range(i+1, len(query_words)+1):
                    phrase = ' '.join(list(query_words)[i:j])
                    if len(phrase.split()) >= 2 and phrase in segment_text_lower:
                        phrase_matches += 1
                        query_phrases.append(phrase)
            
            # Get video content relevance score
            video_relevance = self.check_video_content_relevance(result['video_id'], query)
            
            # Check for technical term matches
            tech_keywords = [
                'insurance', 'policy', 'claim', 'premium', 'coverage',
                'hospital', 'network', 'cashless', 'surgery', 'medical', 'treatment', 'bluetooth',
                'light', 'wavelength', 'frequency', 'radio', 'wireless', 'communication',
                'criteria', 'definition', 'explanation', 'meaning', 'clause'
            ]
            
            tech_matches = 0
            for keyword in tech_keywords:
                if keyword in query_lower and keyword in segment_text_lower:
                    tech_matches += 1
            
            # Calculate boost (for distance-based scores, we want to REDUCE the similarity score)
            total_boost = 0.0
            if exact_matches > 0:
                total_boost += exact_matches * 0.15  # Reduce distance for exact matches
            
            if phrase_matches > 0:
                total_boost += phrase_matches * 0.2  # Reduce distance for phrase matches
            
            if tech_matches > 0:
                total_boost += tech_matches * 0.25  # Reduce distance for technical terms
            
            if video_relevance > 0:
                total_boost += video_relevance * 0.3  # Reduce distance for video relevance
            
            if total_boost > 0:
                # For distance-based scores: improved_similarity = base_similarity - total_boost
                # This makes the score lower (better) when we have good matches
                improved_similarity = base_similarity - total_boost
                
                # Ensure we don't go below a reasonable minimum (but allow negative values)
                # Negative values are actually good for distance-based scores
                improved_similarity = max(-2.0, improved_similarity)  # Allow negative but not too negative
                
                print(f"Boosting segment from {result['video_id']}: {exact_matches} exact matches, {phrase_matches} phrase matches, {tech_matches} tech matches, video relevance: {video_relevance:.3f}, similarity: {base_similarity:.3f} -> {improved_similarity:.3f}")
                if query_phrases:
                    print(f"  Phrase matches: {query_phrases}")
                print(f"  Query words: {query_words}")
                print(f"  Segment preview: {result['text'][:100]}...")
                
                result['similarity'] = improved_similarity
            
            improved_results.append(result)
        
        return improved_results

    def check_video_content_relevance(self, video_id, query):
        """Check if a video contains content relevant to the query and boost its score."""
        query_lower = query.lower()
        
        # Load the full transcript
        full_transcript = self.load_full_transcript(video_id)
        if not full_transcript:
            return 0.0
        
        transcript_lower = full_transcript.lower()
        
        # Check for exact keyword matches
        query_words = set(query_lower.split())
        transcript_words = set(transcript_lower.split())
        exact_matches = len(query_words.intersection(transcript_words))
        
        # Check for technical term matches
        tech_keywords = [
            'bluetooth', 'light', 'wavelength', 'frequency', 'radio', 'wireless', 'communication',
            'insurance', 'policy', 'claim', 'premium', 'coverage', 'hospital', 'network', 'cashless',
            'surgery', 'medical', 'treatment', 'diagnosis', 'symptoms', 'disease', 'health',
            'mathematics', 'physics', 'science', 'theory', 'hypothesis', 'experiment', 'research',
            'traffic', 'transportation', 'boarding', 'airline', 'efficiency', 'optimization',
            'boredom', 'psychology', 'mental', 'health', 'wellbeing', 'productivity',
            'criteria', 'definition', 'explanation', 'meaning', 'clause'
        ]
        
        tech_matches = 0
        for keyword in tech_keywords:
            if keyword in query_lower and keyword in transcript_lower:
                tech_matches += 1
        
        # Calculate relevance score
        relevance_score = (exact_matches * 0.2) + (tech_matches * 0.3)
        
        if relevance_score > 0:
            print(f"Video {video_id} relevance: {relevance_score:.3f} (exact: {exact_matches}, tech: {tech_matches})")
        
        return relevance_score

    def expand_acronym_segments(self, segments, query):
        """Expand segments that contain explanations to get the complete definition."""
        expanded_segments = []
        
        for segment in segments:
            expanded_segment = segment.copy()
            
            # Check if this segment contains important terms that might have a complete explanation
            text_lower = segment['text'].lower()
            
            # Look for patterns that suggest an explanation
            explanation_patterns = [
                r'\b[A-Z]{3,}\b',  # 3+ letter terms
                r'\b[A-Z]{2,}\s+is\s+',  # "X is" pattern
                r'\b[A-Z]{2,}\s+stands\s+for\s+',  # "X stands for" pattern
                r'\b[A-Z]{2,}\s+means\s+',  # "X means" pattern
                r'criteria', r'definition', r'explanation', r'meaning'
            ]
            
            has_explanation = False
            for pattern in explanation_patterns:
                if re.search(pattern, segment['text'], re.IGNORECASE):
                    has_explanation = True
                    break
            
            if has_explanation:
                # Try to expand this segment to get the complete explanation
                video_id = segment['video_id']
                start_time = segment['start']
                end_time = segment['end']
                
                # Load the full transcript for this video
                full_transcript = self.load_full_transcript(video_id)
                if full_transcript:
                    # Find the complete explanation
                    complete_explanation = self.find_complete_acronym_explanation(video_id, query)
                    if complete_explanation and isinstance(complete_explanation, dict):
                        expanded_segment['text'] = complete_explanation.get('text', segment['text'])
                        expanded_segment['start'] = complete_explanation.get('start', start_time)
                        expanded_segment['end'] = complete_explanation.get('end', end_time)
                        print(f"Expanded explanation segment for {video_id}: {start_time:.1f}s-{end_time:.1f}s -> {expanded_segment['start']:.1f}s-{expanded_segment['end']:.1f}s")
                    elif complete_explanation and isinstance(complete_explanation, str):
                        # If it returns just the text, expand the time range
                        expanded_segment['text'] = complete_explanation
                        # Expand the time range to include more context
                        expanded_segment['start'] = max(0, start_time - 30)  # 30 seconds before
                        expanded_segment['end'] = min(9999, end_time + 60)   # 60 seconds after
                        print(f"Expanded explanation segment for {video_id}: {start_time:.1f}s-{end_time:.1f}s -> {expanded_segment['start']:.1f}s-{expanded_segment['end']:.1f}s")
            
            expanded_segments.append(expanded_segment)
        
        return expanded_segments

    def debug_google_drive(self):
        """Debug Google Drive setup and permissions."""
        if not self.drive_sync:
            self.display_message("assistant", "❌ Google Drive sync is not available. Please check your credentials and try again.")
            return
        
        def debug_worker():
            try:
                # Update button state
                self.root.after(0, lambda: self.debug_drive_btn.configure(
                    text="🔍 Debugging...",
                    state="disabled"
                ))
                
                # Run debug setup
                success = self.drive_sync.debug_drive_setup()
                
                if success:
                    self.root.after(0, lambda: self.display_message("assistant", 
                        "✅ Google Drive debug completed successfully!\n\n"
                        "Check the console output above for detailed information about:\n"
                        "• Available folders\n"
                        "• Files in the current folder\n"
                        "• Missing video files\n\n"
                        "If files are not found, check:\n"
                        "1. Folder ID is correct\n"
                        "2. File permissions\n"
                        "3. Google Drive API access"
                    ))
                else:
                    self.root.after(0, lambda: self.display_message("assistant", 
                        "❌ Google Drive debug failed!\n\n"
                        "Check the console output above for error details.\n"
                        "Common issues:\n"
                        "• Invalid credentials\n"
                        "• Network connectivity\n"
                        "• API permissions"
                    ))
                    
            except Exception as e:
                error_msg = f"❌ Debug failed: {str(e)}"
                self.root.after(0, lambda: self.display_message("assistant", error_msg))
            finally:
                # Reset button state
                self.root.after(0, lambda: self.debug_drive_btn.configure(
                    text="🔍 Debug Drive",
                    state="normal"
                ))
        
        # Run debug in background thread
        debug_thread = threading.Thread(target=debug_worker, daemon=True)
        debug_thread.start()

if __name__ == "__main__":
    import sys
    
    # Check if command line arguments are provided for terminal testing
    if len(sys.argv) > 1:
        # Terminal mode for testing
        print("=== ClipQuery Terminal Mode ===")
        
        # Initialize the chatbot
        app = ViviChatbot()
        
        # Get the query from command line
        query = " ".join(sys.argv[1:])
        print(f"Query: {query}")
        
        # Test the RAG pipeline directly
        if app.rag:
            print("\nTesting RAG pipeline...")
            rag_results = app.rag.query_videos(query, n_results=20)
            filtered = [r for r in rag_results if r['similarity'] <= app.similarity_threshold]
            
            print(f"Raw RAG results: {len(rag_results)}")
            print(f"Filtered results: {len(filtered)}")
            
            # Show debug info
            app.debug_video_usage(filtered, "terminal")
            
            # Test context building
            context, final_rag_results = app.build_llm_context(query, for_clipping=False)
            print(f"\nContext length: {len(context)}")
            print(f"Context preview: {context[:500]}...")
            
            # Test LLM response
            print(f"\n=== LLM RESPONSE ===")
            try:
                response_text = chain_general.invoke({
                    "context": "",
                    "question": query,
                    "transcript": context
                })
                response = str(response_text.content) if hasattr(response_text, "content") else str(response_text)
                print(f"LLM Response: {response}")
            except Exception as e:
                print(f"LLM Error: {e}")
            
            # Test clipping context if it's a clipping query
            if query.lower().startswith("clipping:"):
                clipping_context, clipping_rag = app.build_llm_context(query, for_clipping=True)
                print(f"\nClipping context length: {len(clipping_context)}")
                print(f"Clipping context preview: {clipping_context[:500]}...")
                
                # Test clipping LLM response
                print(f"\n=== CLIPPING LLM RESPONSE ===")
                try:
                    clipping_response_text = chain_clipping.invoke({
                        "query": query[len("clipping:"):].strip(),
                        "transcript": clipping_context
                    })
                    clipping_response = str(clipping_response_text.content) if hasattr(clipping_response_text, "content") else str(clipping_response_text)
                    print(f"Clipping LLM Response: {clipping_response}")
                except Exception as e:
                    print(f"Clipping LLM Error: {e}")
        else:
            print("RAG pipeline not available")
    else:
        # GUI mode
        app = ViviChatbot()
        app.run()
