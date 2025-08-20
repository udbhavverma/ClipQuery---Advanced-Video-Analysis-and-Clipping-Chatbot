#!/usr/bin/env python3
"""
Rebuild Vector Store with Acronym Metadata
==========================================

This script rebuilds the vector store with enhanced acronym metadata
for faster acronym-based queries.

Usage:
    python rebuild_with_acronyms.py
"""

import os
import shutil
from rag_pipeline import VideoRAG

def rebuild_vector_store():
    """Rebuild the vector store with acronym metadata."""
    print("🔧 Rebuilding Vector Store with Acronym Metadata")
    print("=" * 50)
    
    # Check if vector store exists
    vector_store_path = "vector_store"
    if os.path.exists(vector_store_path):
        print(f"📁 Found existing vector store at: {vector_store_path}")
        response = input("🗑️ Do you want to delete the existing vector store and rebuild? (y/N): ")
        if response.lower() == 'y':
            print("🗑️ Deleting existing vector store...")
            shutil.rmtree(vector_store_path)
            print("✅ Existing vector store deleted")
        else:
            print("❌ Rebuild cancelled")
            return
    else:
        print("📁 No existing vector store found, creating new one...")
    
    # Check if transcripts exist
    transcripts_folder = "Max Life Videos"
    if not os.path.exists(transcripts_folder):
        print(f"❌ Transcripts folder '{transcripts_folder}' not found!")
        return
    
    txt_files = [f for f in os.listdir(transcripts_folder) if f.endswith('.txt')]
    if not txt_files:
        print(f"❌ No transcript files found in '{transcripts_folder}' folder!")
        return
    
    print(f"📋 Found {len(txt_files)} transcript files to process")
    
    # Initialize and rebuild RAG system
    try:
        print("🚀 Initializing new RAG system with acronym metadata...")
        rag = VideoRAG()
        
        print("✅ Vector store rebuilt successfully!")
        print(f"📊 Processed {len(txt_files)} transcript files")
        
        # Test the new system
        print("\n🧪 Testing acronym search with new metadata...")
        test_query = "What is NOPP?"
        results = rag.query_videos(test_query, n_results=3)
        
        if results:
            print(f"✅ Test successful! Found {len(results)} results for 'NOPP'")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['video_id']} [{result['start']:.2f}-{result['end']:.2f}]")
        else:
            print("⚠️ No results found for test query")
            
    except Exception as e:
        print(f"❌ Error rebuilding vector store: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    rebuild_vector_store() 