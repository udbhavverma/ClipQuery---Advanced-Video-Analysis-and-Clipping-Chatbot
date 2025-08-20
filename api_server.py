import os
import sys
import json
import tempfile
import uuid
import subprocess
import re
import time
from typing import List, Dict, Optional, Union
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import uvicorn

# Import the RAG pipeline
from rag_pipeline import VideoRAG

# Set up environment
os.environ["GROQ_API_KEY"] = "YOUR_API_KEY_HERE"

# Global variables for shared resources
llm = None
llm_clipping = None
rag = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for FastAPI - handles startup and shutdown"""
    global llm, llm_clipping, rag, general_chain, clipping_chain
    
    # Startup
    print("🚀 Starting ViviRAG API Server...")
    print("📚 Initializing LLM models...")
    
    # Initialize LLM models
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=1)
    llm_clipping = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    
    # Create chains after LLM initialization
    print("🔗 Creating LLM chains...")
    general_chain = general_prompt | llm | StrOutputParser()
    clipping_chain = clipping_prompt | llm_clipping | StrOutputParser()
    
    print("🔍 Initializing RAG system...")
    print("⏳ Processing transcript files (this may take a few minutes for large collections)...")
    
    # Initialize RAG system
    start_time = time.time()
    rag = VideoRAG()
    processing_time = time.time() - start_time
    
    print(f"✅ RAG system initialized in {processing_time:.2f} seconds")
    print(f"📊 Loaded {len(rag.vector_store.video_collection.get()['ids'])} videos")
    print("🎯 API Server ready to handle requests!")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down ViviRAG API Server...")
    if rag:
        print("🧹 Cleaning up RAG resources...")
    print("👋 Server shutdown complete")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="ViviRAG API",
    description="AI-powered video analysis and clipping API using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    query: str
    include_clips: bool = False
    max_clips: int = 5

class ClipInfo(BaseModel):
    video_id: str
    start_time: float
    end_time: float
    relevance: str

class QueryResponse(BaseModel):
    response: str
    clips: Optional[List[ClipInfo]] = None
    video_count: int = 0
    processing_time: float = 0.0

class HealthResponse(BaseModel):
    status: str
    rag_loaded: bool
    llm_available: bool

# Templates from original ViviRAG system
general_template = """
You are Vivi, an expert and friendly video assistant chatbot.
You are having an ongoing conversation with the user. You have access to a full transcript of a video. If the user's question is about the video, answer helpfully and refer to timestamps if relevant.
If the question is general and not related to the video, just respond helpfully like a normal assistant.

---
Conversation History:
{context}
---
Full Transcript of the Video:
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

# Create prompt templates
general_prompt = ChatPromptTemplate.from_template(general_template)
clipping_prompt = ChatPromptTemplate.from_template(clipping_template)

# Chains will be created in lifespan function after LLM initialization
general_chain = None
clipping_chain = None

# Utility functions from original ViviRAG system
def estimate_tokens(text):
    """Estimate token count for text."""
    return len(text.split()) * 1.3  # Rough estimate

def check_token_limit(text, max_tokens=8000):
    """Check and truncate text if it exceeds token limit."""
    estimated_tokens = estimate_tokens(text)
    if estimated_tokens <= max_tokens:
        return text
    
    # Truncate to fit within token limit
    target_chars = int(len(text) * (max_tokens / estimated_tokens))
    if target_chars < len(text):
        text = text[:target_chars] + "..."
    
    return text

def load_full_transcript(video_id):
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

def build_llm_context(query, for_clipping=False):
    """Use RAG to find relevant videos/segments and build the transcript context for the LLM."""
    if not rag:
        return "", []

    # --- 1. Extract important terms and handle multi-part queries ---
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
        rag_results = rag.query_videos(subq, n_results=n_results)
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
    
    # Now fill up to 4 with the next best globally (reduced from 8)
    all_rag_results_sorted = sorted(filtered, key=lambda x: x['similarity'])
    for r in all_rag_results_sorted:
        if len(top_video_ids) >= 4:
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
    
    # Determine query type and set appropriate limits
    is_simple_query = len(subqueries) == 1 and len(filtered) <= 3
    is_clipping_query = for_clipping
    is_complex_query = len(subqueries) > 1 or len(filtered) > 6
    
    # Set limits based on query type
    if is_simple_query:
        max_videos = 2
        max_tokens = 4000
        similarity_threshold_for_full = 0.6  # Only include videos with high relevance
    elif is_clipping_query:
        max_videos = 4
        max_tokens = 8000
        similarity_threshold_for_full = 0.5  # Include more videos for clipping accuracy
    else:  # Complex query
        max_videos = 4
        max_tokens = 6000
        similarity_threshold_for_full = 0.55  # Moderate threshold
    
    print(f"Query type: {'Simple' if is_simple_query else 'Clipping' if is_clipping_query else 'Complex'}")
    print(f"Max videos: {max_videos}, Max tokens: {max_tokens}, Similarity threshold: {similarity_threshold_for_full}")
    
    # Filter videos by relevance score before including full transcripts
    relevant_video_ids = []
    for video_id in top_video_ids:
        # Find the best similarity score for this video
        video_segments = [r for r in filtered if r['video_id'] == video_id]
        if video_segments:
            best_similarity = max(r['similarity'] for r in video_segments)
            if best_similarity >= similarity_threshold_for_full:
                relevant_video_ids.append(video_id)
                print(f"Including {video_id} (similarity: {best_similarity:.3f})")
            else:
                print(f"Excluding {video_id} (similarity: {best_similarity:.3f} < {similarity_threshold_for_full})")
        else:
            # If no segments found, include it (might be from guaranteed inclusion)
            relevant_video_ids.append(video_id)
            print(f"Including {video_id} (guaranteed inclusion)")
    
    # ALWAYS include full transcripts for the relevant videos (up to max_videos)
    # This ensures the LLM has complete context for accurate decisions
    videos_added = 0
    for video_id in relevant_video_ids:
        full_transcript = load_full_transcript(video_id)
        if full_transcript:
            # For clipping queries, we need the full transcript for accurate timestamp decisions
            # For normal queries, we also need the full transcript for comprehensive answers
            # No truncation as requested
            
            context_lines.append(f"=== Video: {video_id} ===\n{full_transcript}")
            total_chars += len(full_transcript)
            videos_added += 1
            print(f"Added full transcript for {video_id} ({len(full_transcript)} chars)")
            
            # Stop at max_videos to keep context manageable
            if videos_added >= max_videos:
                print(f"Reached {max_videos} videos limit, stopping transcript inclusion")
                break
    
    final_context = "\n\n".join(context_lines)
    # Apply dynamic token limit based on query type
    final_context = check_token_limit(final_context, max_tokens=max_tokens)
    estimated_tokens = estimate_tokens(final_context)
    print(f"Full transcript context: {len(final_context)} chars, ~{estimated_tokens} tokens")
    print(f"Videos included: {videos_added}")
    
    return final_context, filtered

def parse_llm_clip_response(response):
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

def parse_clips_from_response(response: str) -> List[ClipInfo]:
    """Parse clip information from LLM response."""
    clips = []
    lines = response.strip().split('\n')
    
    current_clip = None
    current_video_id = None
    
    for line in lines:
        line = line.strip()
        
        # Look for video specification
        video_match = re.search(r"-?\s*Video:\s*([^\n]+)", line, re.IGNORECASE)
        if video_match:
            current_video_id = video_match.group(1).strip()
            continue
            
        if line.startswith('- Range: '):
            # Parse timestamp range
            try:
                range_part = line[len('- Range: '):]
                start_str, end_str = range_part.split(' - ')
                start_time = float(start_str)
                end_time = float(end_str)
                
                if start_time < end_time and start_time >= 0:
                    current_clip = {
                        'start_time': start_time,
                        'end_time': end_time,
                        'relevance': '',
                        'video_id': current_video_id or 'video'
                    }
            except (ValueError, IndexError):
                continue
        elif line.startswith('  Relevance: ') and current_clip:
            # Parse relevance explanation
            relevance = line[len('  Relevance: '):]
            current_clip['relevance'] = relevance
            clips.append(ClipInfo(
                video_id=current_clip['video_id'],
                start_time=current_clip['start_time'],
                end_time=current_clip['end_time'],
                relevance=current_clip['relevance']
            ))
            current_clip = None
    
    return clips

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and component availability."""
    try:
        # Check if RAG system and chains are loaded
        if rag is None or general_chain is None or clipping_chain is None:
            return HealthResponse(
                status="initializing",
                rag_loaded=False,
                llm_available=False
            )
        
        # Test RAG
        rag_loaded = rag is not None
        
        # Test LLM with a simple query
        test_response = general_chain.invoke({
            "context": "",
            "transcript": "Test transcript",
            "question": "Hello"
        })
        llm_available = len(test_response) > 0
        
        return HealthResponse(
            status="healthy",
            rag_loaded=rag_loaded,
            llm_available=llm_available
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            rag_loaded=False,
            llm_available=False
        )

@app.get("/status")
async def get_startup_status():
    """Get detailed startup status and processing information."""
    if rag is None:
        return {
            "status": "initializing",
            "message": "RAG system is still being initialized. Please wait...",
            "rag_loaded": False,
            "llm_loaded": llm is not None,
            "video_count": 0,
            "processing_time": 0
        }
    
    try:
        video_count = len(rag.vector_store.video_collection.get()["ids"])
        return {
            "status": "ready",
            "message": "API Server is ready to handle requests",
            "rag_loaded": True,
            "llm_loaded": llm is not None,
            "video_count": video_count,
            "processing_time": "completed"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting status: {str(e)}",
            "rag_loaded": False,
            "llm_loaded": llm is not None,
            "video_count": 0,
            "processing_time": 0
        }

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a general query and return LLM response."""
    import time
    start_time = time.time()
    
    # Check if RAG system and chains are loaded
    if rag is None or general_chain is None or clipping_chain is None:
        raise HTTPException(
            status_code=503, 
            detail="System is still initializing. Please wait a moment and try again."
        )
    
    try:
        # Check if this is a clipping query
        is_clipping_query = request.query.lower().startswith(('clip:', 'clipping:', 'clip '))
        
        if is_clipping_query:
            # Remove clipping prefix
            clean_query = request.query.lower().replace('clip:', '').replace('clipping:', '').replace('clip ', '').strip()
            
            # Build context using sophisticated logic
            transcript_context, rag_results = build_llm_context(clean_query, for_clipping=True)
            
            # Use clipping chain
            response = clipping_chain.invoke({
                "query": clean_query,
                "transcript": transcript_context
            })
            
            # Ensure response is a string
            if response is None:
                response = "No response generated from LLM"
            elif not isinstance(response, str):
                response = str(response)
            
            # Parse clips if requested
            clips = None
            if request.include_clips:
                clips = parse_clips_from_response(response)
                # Limit clips if specified
                if request.max_clips and len(clips) > request.max_clips:
                    clips = clips[:request.max_clips]
        else:
            # Build context using sophisticated logic
            transcript_context, rag_results = build_llm_context(request.query, for_clipping=False)
            
            # Use general chain
            response = general_chain.invoke({
                "context": "",
                "transcript": transcript_context,
                "question": request.query
            })
            
            # Ensure response is a string
            if response is None:
                response = "No response generated from LLM"
            elif not isinstance(response, str):
                response = str(response)
            
            clips = None
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            response=response,
            clips=clips,
            video_count=len(rag.vector_store.video_collection.get()["ids"]) if hasattr(rag, 'vector_store') else 0,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/rag_query", response_model=QueryResponse)
async def process_rag_query(request: QueryRequest):
    """Process a query using RAG pipeline for enhanced accuracy."""
    import time
    start_time = time.time()
    
    # Check if RAG system and chains are loaded
    if rag is None or general_chain is None or clipping_chain is None:
        raise HTTPException(
            status_code=503, 
            detail="System is still initializing. Please wait a moment and try again."
        )
    
    try:
        # Check if this is a clipping query
        is_clipping_query = request.query.lower().startswith(('clip:', 'clipping:', 'clip '))
        
        if is_clipping_query:
            # Remove clipping prefix
            clean_query = request.query.lower().replace('clip:', '').replace('clipping:', '').replace('clip ', '').strip()
        else:
            clean_query = request.query
        
        # Build context using sophisticated logic from original ViviRAG
        transcript_context, rag_results = build_llm_context(clean_query, for_clipping=is_clipping_query)
        
        if is_clipping_query:
            # Use clipping chain with RAG context
            response = clipping_chain.invoke({
                "query": clean_query,
                "transcript": transcript_context
            })
            
            # Ensure response is a string
            if response is None:
                response = "No response generated from LLM"
            elif not isinstance(response, str):
                response = str(response)
            
            # Parse clips if requested
            clips = None
            if request.include_clips:
                clips = parse_clips_from_response(response)
                # Limit clips if specified
                if request.max_clips and len(clips) > request.max_clips:
                    clips = clips[:request.max_clips]
        else:
            # Use general chain with RAG context
            response = general_chain.invoke({
                "context": "",
                "transcript": transcript_context,
                "question": clean_query
            })
            
            # Ensure response is a string
            if response is None:
                response = "No response generated from LLM"
            elif not isinstance(response, str):
                response = str(response)
            
            clips = None
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            response=response,
            clips=clips,
            video_count=len(rag.vector_store.video_collection.get()["ids"]) if hasattr(rag, 'vector_store') else 0,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing RAG query: {str(e)}")

@app.get("/videos")
async def list_available_videos():
    """List all available videos in the RAG system."""
    # Check if RAG system is loaded
    if rag is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG system is still initializing. Please wait a moment and try again."
        )
    
    try:
        if hasattr(rag, 'vector_store'):
            return {"videos": rag.vector_store.video_collection.get()["ids"]}
        else:
            return {"videos": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing videos: {str(e)}")

@app.get("/")
async def root():
    """API root endpoint with basic information."""
    return {
        "message": "ViviRAG API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Process general queries",
            "POST /rag_query": "Process queries with RAG enhancement",
            "GET /health": "Check API health",
            "GET /videos": "List available videos"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 