# ViviRAG API Testing Guide

This guide explains how to use the testing templates for the ViviRAG API server.

## 🚀 Quick Start

### 1. Start the API Server
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Run Quick Tests
```bash
python quick_test.py
```

### 3. Run Comprehensive Tests
```bash
python test_api_client.py
```

### 4. Performance Testing
```bash
python quick_lifespan_test.py
```

### 5. Debug Testing
```bash
python debug_test.py
```

## 📋 Testing Templates

### 1. `quick_test.py` - Quick Testing
**Purpose**: Rapid testing of common queries
**Features**:
- Tests 5 common queries
- Shows response time and preview
- Displays generated clips
- Simple and fast

**Usage**:
```bash
python quick_test.py
```

**Test Queries**:
- "What is NOPP?"
- "What is Star ULIP?"
- "clip: What is NOPP?"
- "What is NOPP and Star ULIP?"
- "clip: What is NOPP and Star ULIP?"

### 2. `test_api_client.py` - Comprehensive Testing
**Purpose**: Thorough testing with multiple scenarios
**Features**:
- 15+ test cases
- Performance metrics
- Success/failure tracking
- Detailed reporting

**Usage**:
```bash
# Run all tests
python test_api_client.py
```

**Test Categories**:
- ✅ Basic Health and System Tests
- 🔍 General Query Tests
- 🔗 Multi-Topic Query Tests
- 🎬 Clipping Query Tests
- 🧠 Complex Query Tests
- ⚡ Edge Case Tests
- 📊 Performance Tests

### 3. `quick_lifespan_test.py` - Performance Testing
**Purpose**: Test API performance and lifespan optimization
**Features**:
- Tests startup time
- Measures response times
- Validates lifespan events
- Performance benchmarking

**Usage**:
```bash
python quick_lifespan_test.py
```

### 4. `debug_test.py` - Debug Testing
**Purpose**: Debug API responses and troubleshoot issues
**Features**:
- Detailed response analysis
- Error debugging
- Response structure validation
- API state inspection

**Usage**:
```bash
python debug_test.py
```

## 🎮 Interactive Testing

You can also test the API interactively using curl commands:

```bash
# Health check
curl http://localhost:8000/health

# List videos
curl http://localhost:8000/videos

# Test a query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is NOPP?", "include_clips": false}'

# Test clipping
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "clip: What is NOPP?", "include_clips": true, "max_clips": 3}'
```

## 📊 Test Results Interpretation

### Success Indicators
- ✅ **PASS**: Query processed successfully
- ⚡ **FAST**: Response time under 5 seconds
- 📝 **Response Length**: Number of characters in response
- 🎬 **Clips**: Number of video clips generated
- 📹 **Videos**: Number of videos available in system

### Performance Metrics
- **Fast**: < 5 seconds
- **Slow**: > 5 seconds
- **Success Rate**: Percentage of passed tests

## 🔧 Custom Testing

### Creating Custom Test Cases

You can modify `test_api_client.py` to add your own test cases:

```python
# Add to test_api_client.py
def test_custom_query():
    """Test a custom query"""
    response = requests.post(
        "http://localhost:8000/query",
        json={
            "query": "Your custom query here",
            "include_clips": True,
            "max_clips": 3
        }
    )
    assert response.status_code == 200
    result = response.json()
    print(f"Response: {result['response'][:100]}...")
```

### Testing Different Query Types

1. **General Queries**: `"What is NOPP?"`
2. **Clipping Queries**: `"clip: What is NOPP?"`
3. **Multi-Topic Queries**: `"What is NOPP and Star ULIP?"`
4. **Complex Queries**: `"What are the benefits of term life insurance and how does it compare to ULIP?"`

## 🐛 Troubleshooting

### Common Issues

1. **API Server Not Running**
   ```
   ❌ Error: Connection refused
   ```
   **Solution**: Start the API server with `uvicorn api_server:app --reload`

2. **System Still Initializing**
   ```
   ❌ Error: System is still initializing
   ```
   **Solution**: Wait for startup completion, check `/status` endpoint

3. **No Response Text**
   ```
   📝 Response Length: 0 chars
   ```
   **Solution**: Check if the query is valid and the API is processing correctly

4. **Slow Response Times**
   ```
   ⏱️ SLOW | Test Name (25.23s)
   ```
   **Solution**: This is normal for complex queries with multiple videos

### Debug Commands

```bash
# Check API health
curl http://localhost:8000/health

# Check startup status
curl http://localhost:8000/status

# List available videos
curl http://localhost:8000/videos

# Test a specific query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is NOPP?", "include_clips": false}'
```

## 📈 Performance Benchmarks

### Expected Response Times
- **Simple Queries**: 1-3 seconds
- **Complex Queries**: 3-8 seconds
- **Multi-Topic Queries**: 5-15 seconds
- **Clipping Queries**: 3-10 seconds

### Success Rate Targets
- **Health Check**: 100%
- **Basic Queries**: >90%
- **Complex Queries**: >80%
- **Overall**: >85%

## 🎯 Best Practices

1. **Start with Quick Tests**: Use `quick_test.py` for rapid validation
2. **Use Debug Tests**: Use `debug_test.py` for troubleshooting
3. **Monitor Performance**: Watch response times and success rates
4. **Test Edge Cases**: Include minimal and verbose queries
5. **Validate Clips**: Check that clipping queries generate proper timestamps
6. **Check Startup**: Ensure lifespan optimization is working

## 📝 Example Test Scenarios

### Scenario 1: Basic Functionality
```bash
python quick_test.py
```

### Scenario 2: Comprehensive Testing
```bash
python test_api_client.py
```

### Scenario 3: Performance Testing
```bash
python quick_lifespan_test.py
```

### Scenario 4: Debug Testing
```bash
python debug_test.py
```

### Scenario 5: Custom Query Testing
```python
# Add to test_api_client.py
def test_custom_insurance_query():
    """Test specific insurance comparison"""
    response = requests.post(
        "http://localhost:8000/query",
        json={
            "query": "What is the difference between term and whole life insurance?",
            "include_clips": False
        }
    )
    assert response.status_code == 200
    result = response.json()
    print(f"Response: {result['response'][:100]}...")
```

## 🔍 Advanced Testing

### Testing Lifespan Optimization
```bash
# Test startup time
python quick_lifespan_test.py

# Check if transcripts are processed only once
curl http://localhost:8000/status
```

### Testing Token Optimization
```bash
# Test simple query (should use fewer tokens)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is NOPP?", "include_clips": false}'

# Test complex query (should use more tokens but still optimized)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is NOPP and explain Star ULIP and health insurance benefits", "include_clips": false}'
```

### Testing Acronym Detection
```bash
# Test acronym query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is NOPP?", "include_clips": false}'

# Test multi-acronym query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain Star ULIP", "include_clips": false}'
```

This testing framework provides comprehensive coverage of the ViviRAG API functionality and helps ensure reliable performance across different query types and scenarios. 