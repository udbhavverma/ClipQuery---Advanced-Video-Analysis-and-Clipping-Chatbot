import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Any
import json
from datetime import datetime
import re

class VectorStore:
    def __init__(self, persist_directory: str = "vector_store"):
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Create or get collections
        self.video_collection = self.client.get_or_create_collection(
            name="videos",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.segment_collection = self.client.get_or_create_collection(
            name="segments",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Load embeddings model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.load_transcripts()
        
    def load_transcripts(self):
        """Load and process transcript files from Max Life Videos folder"""
        transcripts_folder = "Max Life Videos"
        
        if not os.path.exists(transcripts_folder):
            print(f"Transcripts folder '{transcripts_folder}' not found!")
            return
            
        # Get all .txt files in the folder
        txt_files = [f for f in os.listdir(transcripts_folder) if f.endswith('.txt')]
        
        if not txt_files:
            print(f"No transcript files found in '{transcripts_folder}' folder!")
            return
            
        print(f"Found {len(txt_files)} transcript files")
        
        for txt_file in txt_files:
            video_name = txt_file.replace('.txt', '')
            file_path = os.path.join(transcripts_folder, txt_file)
            
            print(f"Processing {video_name}...")
            
            # Add video to collection if not exists
            if not self.video_collection.get(ids=[video_name])["ids"]:
                self.video_collection.add(
                    ids=[video_name],
                    documents=[video_name],
                    metadatas=[{
                        "name": video_name,
                        "processed_at": datetime.now().isoformat()
                    }]
                )
            
            # Process transcript file
            self.process_transcript_file(file_path, video_name)
            
        print("Transcript processing completed!")
        
    def extract_acronyms_from_text(self, text: str) -> List[str]:
        """Extract acronyms from text using improved detection logic"""
        # Convert to uppercase for consistent matching
        text_upper = text.upper()
        
        # Extract all-uppercase words of length 3+ (including those with punctuation)
        acronym_pattern = re.findall(r'\b[A-Z]{3,}\b', text_upper)
        
        # Exclude common words and insurance terms
        common_words = {
            'WHAT', 'WHEN', 'WHERE', 'WHY', 'HOW', 'THE', 'AND', 'FOR', 'ARE', 'YOU', 'YOUR', 'THEY', 'THEIR',
            'WITH', 'FROM', 'THAT', 'THIS', 'HAVE', 'HAS', 'HAD', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'MIGHT',
            'MUST', 'CAN', 'MAY', 'GIVE', 'FULL', 'EXPLANATION', 'MEANING', 'DEFINITION', 'STANDS', 'ABOUT',
            # Insurance/finance terms to exclude as acronyms
            'CLAUSE', 'POLICY', 'PLAN', 'COVER', 'TERM', 'LIFE', 'RIDER', 'BENEFIT', 'SUM', 'ASSURED', 'PREMIUM',
            'HEALTH', 'INSURANCE', 'OPTION', 'FEATURE', 'VALUE', 'EXIT', 'VARIANT', 'YEAR', 'AGE', 'COST', 'TAX',
            'LOAN', 'BANK', 'CSR', 'ICICI', 'CS', 'IRDAI', 'CSR', 'CS', 'SBI', 'TATA', 'PLUS', 'MAX', 'ALLIANCE',
            'SMART', 'REGULAR', 'SHIELD', 'ACCIDENTAL', 'RETURN', 'PREMIUMS', 'DISCOUNT', 'INCOME', 'PROTECT',
            'FAMILY', 'PROTECTION', 'WAIVER', 'CRITICAL', 'ILLNESS', 'DEATH', 'SETTLEMENT', 'RATIO', 'SOLVENCY',
            'COMPLAINTS', 'REVIEW', 'SUMMARY', 'SUMMARY', 'SUMMARY', 'SUMMARY', 'SUMMARY', 'SUMMARY', 'SUMMARY'
        }
        
        # Only keep true acronyms
        acronyms = [a for a in acronym_pattern if a not in common_words]
        
        return acronyms

    def process_transcript_file(self, file_path: str, video_name: str):
        """Process a single transcript file and extract segments with acronym metadata"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split content into lines and process each line
            lines = content.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse timestamp and text using regex
                match = re.match(r'\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s*(.+)', line)
                if match:
                    start_time = float(match.group(1))
                    end_time = float(match.group(2))
                    text = match.group(3).strip()
                    
                    if text:  # Only process if there's actual text content
                        # Generate embedding for the text
                        embedding = self.embedding_model.encode(text)
                        
                        # Extract acronyms from this segment
                        acronyms = self.extract_acronyms_from_text(text)
                        
                        # Create segment ID
                        segment_id = f"{video_name}_{start_time}_{end_time}"
                        
                        # Format timestamp for display
                        timestamp = f"[{start_time:.2f} - {end_time:.2f}]"
                        
                        # Prepare metadata with acronym information
                        metadata = {
                            "video_id": video_name,
                            "start": start_time,
                            "end": end_time,
                            "timestamp": timestamp,
                            "acronyms": ",".join(acronyms) if acronyms else "",  # Store acronyms as comma-separated string
                            "acronym_count": len(acronyms)  # Store count for quick filtering
                        }
                        
                        # Add segment to collection with acronym metadata
                        self.segment_collection.add(
                            ids=[segment_id],
                            documents=[text],
                            embeddings=[embedding.tolist()],
                            metadatas=[metadata]
                        )
                        
                        # Log if acronyms were found
                        if acronyms:
                            print(f"  Found acronyms in {video_name} [{start_time:.2f}-{end_time:.2f}]: {acronyms}")
                        
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            raise
                        
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            raise
        
    def search_segments(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant video segments"""
        # Improved acronym detection: only all-uppercase, 3+ chars, not common/insurance words
        query_upper = query.upper()
        # Extract all-uppercase words of length 3+ (including those with punctuation)
        acronym_pattern = re.findall(r'\b[A-Z]{3,}\b', query_upper)
        # Exclude common words and insurance terms
        common_words = {
            'WHAT', 'WHEN', 'WHERE', 'WHY', 'HOW', 'THE', 'AND', 'FOR', 'ARE', 'YOU', 'YOUR', 'THEY', 'THEIR',
            'WITH', 'FROM', 'THAT', 'THIS', 'HAVE', 'HAS', 'HAD', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'MIGHT',
            'MUST', 'CAN', 'MAY', 'GIVE', 'FULL', 'EXPLANATION', 'MEANING', 'DEFINITION', 'STANDS', 'ABOUT',
            # Insurance/finance terms to exclude as acronyms
            'CLAUSE', 'POLICY', 'PLAN', 'COVER', 'TERM', 'LIFE', 'RIDER', 'BENEFIT', 'SUM', 'ASSURED', 'PREMIUM',
            'HEALTH', 'INSURANCE', 'OPTION', 'FEATURE', 'VALUE', 'EXIT', 'VARIANT', 'YEAR', 'AGE', 'COST', 'TAX',
            'LOAN', 'BANK', 'CSR', 'ICICI', 'CS', 'IRDAI', 'CSR', 'CS', 'SBI', 'TATA', 'PLUS', 'MAX', 'ALLIANCE',
            'SMART', 'REGULAR', 'SHIELD', 'ACCIDENTAL', 'RETURN', 'PREMIUMS', 'DISCOUNT', 'INCOME', 'PROTECT',
            'FAMILY', 'PROTECTION', 'WAIVER', 'CRITICAL', 'ILLNESS', 'DEATH', 'SETTLEMENT', 'RATIO', 'SOLVENCY',
            'COMPLAINTS', 'REVIEW', 'SUMMARY', 'SUMMARY', 'SUMMARY', 'SUMMARY', 'SUMMARY', 'SUMMARY', 'SUMMARY'
        }
        # Only keep true acronyms
        acronyms = [a for a in acronym_pattern if a not in common_words]
        
        # Treat as acronym query if:
        # 1. Query contains explanation keywords, OR
        # 2. Query is just an acronym (like "NOPP?")
        is_explanation_query = any(keyword in query_upper for keyword in ['EXPLAIN', 'MEANING', 'DEFINITION', 'STANDS', 'WHAT IS', 'TELL ME ABOUT'])
        is_standalone_acronym = len(acronyms) == 1 and len(query.strip()) <= 10  # Short query with just an acronym
        
        if acronyms and (is_explanation_query or is_standalone_acronym):
            print(f"Acronym explanation query detected: {acronyms}")
            # For multi-acronym queries, search and merge results for each
            all_results = []
            for acronym in acronyms:
                results = self._search_acronym_segments([acronym], n_results)
                all_results.extend(results)
            # Remove duplicates by segment id
            seen = set()
            unique_results = []
            for r in all_results:
                seg_id = (r['video_id'], r['start'], r['end'])
                if seg_id not in seen:
                    unique_results.append(r)
                    seen.add(seg_id)
            # Sort by score (higher is better)
            unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            return unique_results[:n_results]
        else:
            print(f"Regular semantic query detected")
            return self._search_semantic_segments(query, n_results)
    
    def _search_acronym_segments(self, acronyms: List[str], n_results: int) -> List[Dict[str, Any]]:
        """Search for segments containing specific acronyms using metadata for faster filtering"""
        print(f"🔍 Searching for acronyms using metadata: {acronyms}")
        
        # First, try to use metadata filtering for faster results
        try:
            # Get all segments that have acronyms in their metadata
            all_results = self.segment_collection.get()
            if not all_results["ids"]:
                return []
            
            scored_segments = []
            for i in range(len(all_results["ids"])):
                metadata = all_results["metadatas"][i]
                segment_text = all_results["documents"][i]
                
                # Check if this segment has any of the requested acronyms in its metadata
                segment_acronyms_str = metadata.get("acronyms", "")
                segment_acronyms = segment_acronyms_str.split(",") if segment_acronyms_str else []
                matching_acronyms = []
                
                for acronym in acronyms:
                    # Check if acronym exists in segment's metadata
                    if acronym in segment_acronyms:
                        matching_acronyms.append(acronym)
                
                # If we found matching acronyms in metadata, process this segment
                if matching_acronyms:
                    # Double-check with text content for accuracy
                    segment_text_upper = segment_text.upper()
                    verified_matches = []
                    
                    for acronym in matching_acronyms:
                        # Normalize: remove spaces/hyphens for matching
                        norm_acronym = acronym.replace(' ', '').replace('-', '').upper()
                        norm_text = segment_text_upper.replace(' ', '').replace('-', '')
                        # Use simple substring matching since we've already normalized
                        if norm_acronym in norm_text:
                            verified_matches.append(acronym)
                    
                    if verified_matches:
                        # Calculate score based on matches and content quality
                        score = len(verified_matches) * 0.5 + (len(segment_text) / 1000) * 0.3
                        
                        # Bonus for multiple requested acronyms in one segment
                        if len(verified_matches) > 1:
                            score += 0.2
                        
                        # Bonus for explanation keywords
                        explanation_keywords = ['MEANS', 'STANDS', 'REFERS', 'DEFINED', 'EXPLAIN', 'DESCRIBE', 'ABOUT', 'CRITERIA', 'NEED', 'OPPORTUNITY', 'PHYSICALLY', 'PAYING', 'CAPACITY']
                        keyword_bonus = sum(0.1 for keyword in explanation_keywords if keyword in segment_text_upper)
                        score += keyword_bonus
                        
                        print(f"Found acronym segment: {segment_text[:100]}... (acronyms: {verified_matches}, score: {score:.3f})")
                        scored_segments.append({
                            "video_id": metadata["video_id"],
                            "text": segment_text,
                            "start": metadata["start"],
                            "end": metadata["end"],
                            "timestamp": metadata["timestamp"],
                            "similarity": 1.0 - score,  # Lower is better
                            "exact_matches": verified_matches,
                            "score": score
                        })
            
            # Sort by score (higher is better)
            scored_segments.sort(key=lambda x: x["score"], reverse=True)
            print(f"Found {len(scored_segments)} segments containing acronyms: {acronyms}")
            
            # If no acronym segments found, fall back to semantic search
            if not scored_segments:
                print(f"No segments found containing acronyms {acronyms}, falling back to semantic search")
                return self._search_semantic_segments(f"What is {' '.join(acronyms)}?", n_results)
            
            return scored_segments[:n_results]
            
        except Exception as e:
            print(f"Error in metadata-based acronym search: {e}")
            # Fall back to original method if metadata search fails
            print("Falling back to original acronym search method")
            return self._search_acronym_segments_fallback(acronyms, n_results)
    
    def _search_acronym_segments_fallback(self, acronyms: List[str], n_results: int) -> List[Dict[str, Any]]:
        """Fallback method for acronym search (original implementation)"""
        all_results = self.segment_collection.get()
        if not all_results["ids"]:
            return []
        scored_segments = []
        for i in range(len(all_results["ids"])):
            segment_text = all_results["documents"][i]
            segment_text_upper = segment_text.upper()
            # Only boost for exact acronym matches (case-insensitive, ignore spaces/hyphens)
            exact_matches = []
            for acronym in acronyms:
                # Normalize: remove spaces/hyphens for matching
                norm_acronym = acronym.replace(' ', '').replace('-', '').upper()
                norm_text = segment_text_upper.replace(' ', '').replace('-', '')
                # Use simple substring matching since we've already normalized
                if norm_acronym in norm_text:
                    exact_matches.append(acronym)
            # Only boost if there is an exact match
            if exact_matches:
                score = len(exact_matches) * 0.5 + (len(segment_text) / 1000) * 0.3
                # Bonus for multiple requested acronyms in one segment
                if len(exact_matches) > 1:
                    score += 0.2
                # Bonus for explanation keywords
                explanation_keywords = ['MEANS', 'STANDS', 'REFERS', 'DEFINED', 'EXPLAIN', 'DESCRIBE', 'ABOUT', 'CRITERIA', 'NEED', 'OPPORTUNITY', 'PHYSICALLY', 'PAYING', 'CAPACITY']
                keyword_bonus = sum(0.1 for keyword in explanation_keywords if keyword in segment_text_upper)
                score += keyword_bonus
                print(f"Found acronym segment: {segment_text[:100]}... (acronyms: {exact_matches}, score: {score:.3f})")
                scored_segments.append({
                    "video_id": all_results["metadatas"][i]["video_id"],
                    "text": segment_text,
                    "start": all_results["metadatas"][i]["start"],
                    "end": all_results["metadatas"][i]["end"],
                    "timestamp": all_results["metadatas"][i]["timestamp"],
                    "similarity": 1.0 - score,  # Lower is better
                    "exact_matches": exact_matches,
                    "score": score
                })
        # Sort by score (higher is better)
        scored_segments.sort(key=lambda x: x["score"], reverse=True)
        print(f"Found {len(scored_segments)} segments containing acronyms: {acronyms}")
        # If no acronym segments found, fall back to semantic search
        if not scored_segments:
            print(f"No segments found containing acronyms {acronyms}, falling back to semantic search")
            return self._search_semantic_segments(f"What is {' '.join(acronyms)}?", n_results)
        return scored_segments[:n_results]
    
    def _search_semantic_segments(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """Search for segments using semantic similarity (original method)"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Search segments
        results = self.segment_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "video_id": results["metadatas"][0][i]["video_id"],
                "text": results["documents"][0][i],
                "start": results["metadatas"][0][i]["start"],
                "end": results["metadatas"][0][i]["end"],
                "timestamp": results["metadatas"][0][i]["timestamp"],
                "similarity": results["distances"][0][i]
            })
            
        return formatted_results

class VideoRAG:
    def __init__(self, vector_store_path: str = "vector_store"):
        self.vector_store = VectorStore(vector_store_path)
        
    def query_videos(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Query the video database"""
        return self.vector_store.search_segments(query, n_results) 