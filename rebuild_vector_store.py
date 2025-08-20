#!/usr/bin/env python3
"""
Script to rebuild the vector store with all transcript data.
"""

import os
import shutil
from rag_pipeline import VideoRAG

def rebuild_vector_store():
    """Rebuild the vector store from scratch."""
    print("🔄 Rebuilding vector store...")
    
    # Remove existing vector store if it exists
    if os.path.exists("vector_store"):
        print("🗑️  Removing existing vector store...")
        shutil.rmtree("vector_store")
    
    # Create new vector store
    print("🏗️  Creating new vector store...")
    rag = VideoRAG()
    
    # Test the search
    print("🔍 Testing NOPP search...")
    results = rag.query_videos("What is NOPP?", 5)
    print(f"Found {len(results)} NOPP results:")
    for r in results:
        print(f"  {r['video_id']}: {r['text'][:80]}...")
    
    print("🔍 Testing STARULIP search...")
    results = rag.query_videos("What is STARULIP?", 5)
    print(f"Found {len(results)} STARULIP results:")
    for r in results:
        print(f"  {r['video_id']}: {r['text'][:80]}...")
    
    print("✅ Vector store rebuilt successfully!")

if __name__ == "__main__":
    rebuild_vector_store() 