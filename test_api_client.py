import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    print("🔍 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health check passed: {data}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False

def test_general_query():
    """Test a general query endpoint."""
    print("\n🔍 Testing general query...")
    
    payload = {
        "query": "What is NOPP?",
        "include_clips": False
    }
    
    response = requests.post(f"{BASE_URL}/query", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ General query response:")
        print(f"Response: {data['response'][:200]}...")
        print(f"Processing time: {data['processing_time']:.2f}s")
        print(f"Video count: {data['video_count']}")
        return True
    else:
        print(f"❌ General query failed: {response.status_code}")
        print(f"Error: {response.text}")
        return False

def test_rag_query():
    """Test a RAG-enhanced query endpoint."""
    print("\n🔍 Testing RAG query...")
    
    payload = {
        "query": "What is NOPP?",
        "include_clips": False
    }
    
    response = requests.post(f"{BASE_URL}/rag_query", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ RAG query response:")
        print(f"Response: {data['response'][:200]}...")
        print(f"Processing time: {data['processing_time']:.2f}s")
        print(f"Video count: {data['video_count']}")
        return True
    else:
        print(f"❌ RAG query failed: {response.status_code}")
        print(f"Error: {response.text}")
        return False

def test_clipping_query():
    """Test a clipping query endpoint."""
    print("\n🔍 Testing clipping query...")
    
    payload = {
        "query": "clip: What is NOPP?",
        "include_clips": True,
        "max_clips": 3
    }
    
    response = requests.post(f"{BASE_URL}/rag_query", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Clipping query response:")
        print(f"Response: {data['response'][:200]}...")
        print(f"Processing time: {data['processing_time']:.2f}s")
        print(f"Video count: {data['video_count']}")
        
        if data['clips']:
            print(f"Clips found: {len(data['clips'])}")
            for i, clip in enumerate(data['clips']):
                print(f"  Clip {i+1}: {clip['start_time']:.2f}s - {clip['end_time']:.2f}s")
                print(f"    Relevance: {clip['relevance']}")
        return True
    else:
        print(f"❌ Clipping query failed: {response.status_code}")
        print(f"Error: {response.text}")
        return False

def test_multi_topic_query():
    """Test a multi-topic query."""
    print("\n🔍 Testing multi-topic query...")
    
    payload = {
        "query": "What is NOPP and Star ULIP?",
        "include_clips": False
    }
    
    response = requests.post(f"{BASE_URL}/rag_query", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Multi-topic query response:")
        print(f"Response: {data['response'][:300]}...")
        print(f"Processing time: {data['processing_time']:.2f}s")
        print(f"Video count: {data['video_count']}")
        return True
    else:
        print(f"❌ Multi-topic query failed: {response.status_code}")
        print(f"Error: {response.text}")
        return False

def test_available_videos():
    """Test the videos endpoint."""
    print("\n🔍 Testing available videos...")
    
    response = requests.get(f"{BASE_URL}/videos")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Available videos: {len(data['videos'])}")
        for video in data['videos'][:5]:  # Show first 5
            print(f"  - {video}")
        return True
    else:
        print(f"❌ Videos endpoint failed: {response.status_code}")
        return False

def main():
    """Run all API tests."""
    print("🚀 Starting ViviRAG API Tests")
    print("=" * 50)
    
    tests = [
        test_health,
        test_available_videos,
        test_general_query,
        test_rag_query,
        test_clipping_query,
        test_multi_topic_query
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the API server.")

if __name__ == "__main__":
    main() 