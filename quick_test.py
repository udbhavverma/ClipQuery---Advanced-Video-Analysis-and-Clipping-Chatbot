#!/usr/bin/env python3
"""
Quick ViviRAG API Test
======================

A simple script for quick API testing.

Usage:
    python quick_test.py
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def quick_test():
    """Run a quick test of the API."""
    print("🚀 Quick ViviRAG API Test")
    print("=" * 40)
    
    # Test queries
    test_queries = [
        "What is NOPP?",
        "What is Star ULIP?",
        "clip: What is NOPP?",
        "What is NOPP and Star ULIP?",
        "clip: What is NOPP and Star ULIP?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 40)
        
        payload = {
            "query": query,
            "include_clips": query.lower().startswith(('clip:', 'clipping:', 'clip ')),
            "max_clips": 3
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(f"{API_BASE_URL}/rag_query", json=payload)
            result = response.json()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"⏱️ Time: {processing_time:.2f}s")
            
            # Safely get response text
            response_text = result.get('response')
            if response_text is None:
                response_length = 0
                print(f"📝 Response Length: 0 chars (None response)")
            else:
                response_length = len(str(response_text))
                print(f"📝 Response Length: {response_length} chars")
            
            clips = result.get('clips')
            clips_count = len(clips) if clips is not None else 0
            print(f"🎬 Clips: {clips_count}")
            
            # Show response preview
            if response_text and response_text is not None and len(str(response_text)) > 0:
                preview = str(response_text)[:150] + "..." if len(str(response_text)) > 150 else str(response_text)
                print(f"📄 Preview: {preview}")
            else:
                print(f"📄 Preview: No response text")
            
            # Show clips if any
            clips = result.get('clips')
            if clips and len(clips) > 0:
                print(f"🎬 Clips:")
                for j, clip in enumerate(clips, 1):
                    print(f"  {j}. {clip['start_time']:.2f}s - {clip['end_time']:.2f}s")
                    print(f"     Video: {clip['video_id']}")
            
            print("✅ Success")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Quick test completed!")

if __name__ == "__main__":
    quick_test() 