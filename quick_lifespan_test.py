#!/usr/bin/env python3
import requests
import time

def test_performance():
    print("⚡ Quick Lifespan Performance Test")
    print("=" * 40)
    
    # Test query
    query = "What is NOPP?"
    
    print(f"🔍 Testing query: {query}")
    
    # Measure response time
    start_time = time.time()
    response = requests.post(
        "http://localhost:8000/query",
        json={"query": query, "include_clips": False}
    )
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Response time: {end_time - start_time:.2f} seconds")
        print(f"📝 Response length: {len(result['response'])} characters")
        print(f"🎯 Processing time: {result['processing_time']:.2f} seconds")
        print(f"📊 Video count: {result['video_count']}")
        print(f"\n📄 Response preview:")
        print(f"{result['response'][:300]}...")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_performance() 