#!/usr/bin/env python3
"""
Debug Test for ViviRAG API
==========================

Simple script to debug API response issues.
"""

import requests
import json

API_BASE_URL = "http://localhost:8000"

def debug_api_response():
    """Debug the API response structure."""
    print("🔍 Debugging API Response Structure")
    print("=" * 50)
    
    # Test query
    query = "What is NOPP?"
    payload = {
        "query": query,
        "include_clips": False,
        "max_clips": 5
    }
    
    try:
        print(f"📤 Sending query: {query}")
        response = requests.post(f"{API_BASE_URL}/rag_query", json=payload)
        
        print(f"📥 Response Status: {response.status_code}")
        print(f"📥 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"📋 Response Keys: {list(result.keys())}")
            
            # Check each field
            for key, value in result.items():
                print(f"  {key}: {type(value)} = {repr(value)}")
                
                if key == 'response':
                    if value is None:
                        print(f"    ❌ Response is None!")
                    elif isinstance(value, str):
                        print(f"    ✅ Response is string, length: {len(value)}")
                        print(f"    📄 Preview: {value[:100]}...")
                    else:
                        print(f"    ⚠️ Response is {type(value)}: {value}")
            
            # Test the specific issue
            response_text = result.get('response', '')
            print(f"\n🔍 Testing response length calculation:")
            print(f"  response_text = {repr(response_text)}")
            print(f"  type(response_text) = {type(response_text)}")
            print(f"  response_text is None = {response_text is None}")
            
            if response_text is None:
                print(f"  ❌ Cannot get length of None")
            else:
                print(f"  ✅ Length: {len(response_text)}")
                
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    debug_api_response() 