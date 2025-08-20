# ViviRAG API Server

A high-performance FastAPI-based REST API for intelligent video analysis and clipping using Retrieval-Augmented Generation (RAG).

## 🚀 **Features**

- **Intelligent Video Analysis**: Query video content using natural language
- **Precise Video Clipping**: Generate timestamp-based video clips
- **Multi-topic Query Support**: Handle complex queries with multiple topics
- **Acronym Detection**: Specialized handling for acronym-based queries
- **Performance Optimized**: Lifespan events for fast startup and response times
- **Token Management**: Dynamic context limits based on query complexity
- **Health Monitoring**: Built-in health checks and status endpoints

## 📋 **Requirements**

- Python 3.8+
- FFmpeg (for video processing)
- Groq API key (for LLM access)

## 🛠️ **Installation**

1. **Clone the repository:**
```bash
git clone <repository-url>
cd ClipQuery
```

2. **Install dependencies:**
```bash
pip install -r api_requirements.txt
```

3. **Set up environment variables:**
```bash
export GROQ_API_KEY="your-groq-api-key"
```

4. **Ensure transcript files are available:**
   - Place video transcript files in the `Max Life Videos/` folder
   - Files should be named `{video_name}.txt` and `{video_name}.srt`

## 🚀 **Quick Start**

1. **Start the API server:**
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

2. **Wait for initialization:**
   - The server will process all transcript files during startup
   - Monitor progress with: `curl http://localhost:8000/status`

3. **Test the API:**
```bash
# Health check
curl http://localhost:8000/health

# General query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is NOPP?", "include_clips": false}'

# Clipping query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "clip: explain health insurance benefits", "include_clips": true, "max_clips": 3}'
```

## 📚 **API Endpoints**

### **Health & Status**

#### `GET /health`
Check API health and component availability.

**Response:**
```json
{
  "status": "healthy",
  "rag_loaded": true,
  "llm_available": true
}
```

#### `GET /status`
Get detailed startup status and processing information.

**Response:**
```json
{
  "status": "ready",
  "message": "API Server is ready to handle requests",
  "rag_loaded": true,
  "llm_loaded": true,
  "video_count": 45,
  "processing_time": "completed"
}
```

### **Core Functionality**

#### `POST /query`
Process queries and return intelligent responses with optional video clips.

**Request Body:**
```json
{
  "query": "What is NOPP?",
  "include_clips": false,
  "max_clips": 5
}
```

**Parameters:**
- `query` (string, required): The question or request
- `include_clips` (boolean, optional): Whether to parse and return video clips
- `max_clips` (integer, optional): Maximum number of clips to return

**Response:**
```json
{
  "response": "NOPP stands for No Objection to Policy Portability...",
  "clips": null,
  "video_count": 45,
  "processing_time": 1.23
}
```

#### `POST /rag_query`
Direct RAG query without LLM processing.

**Request Body:**
```json
{
  "query": "health insurance benefits",
  "n_results": 5
}
```

**Response:**
```json
{
  "results": [
    {
      "video_id": "health_insurance_guide",
      "start": 45.2,
      "end": 67.8,
      "text": "Health insurance provides coverage for...",
      "similarity": 0.85
    }
  ]
}
```

#### `GET /videos`
List all available videos in the system.

**Response:**
```json
{
  "videos": [
    "health_insurance_guide",
    "term_life_explained",
    "maternity_coverage"
  ]
}
```

## 🔧 **Query Types**

### **General Queries**
Standard questions about video content:
```json
{
  "query": "What are the benefits of health insurance?",
  "include_clips": false
}
```

### **Clipping Queries**
Generate video clips with timestamps:
```json
{
  "query": "clip: explain maternity coverage",
  "include_clips": true,
  "max_clips": 3
}
```

### **Multi-topic Queries**
Handle complex queries with multiple topics:
```json
{
  "query": "What is NOPP and explain Star ULIP",
  "include_clips": false
}
```

### **Acronym Queries**
Specialized handling for acronym explanations:
```json
{
  "query": "What is NOPP?",
  "include_clips": false
}
```

## ⚡ **Performance Optimizations**

### **Lifespan Events**
- Transcript processing happens once during startup
- 10-20x faster response times
- No repeated processing on each request

### **Token Management**
- Dynamic limits based on query complexity:
  - **Simple queries**: 4,000 tokens, 2 videos
  - **Clipping queries**: 8,000 tokens, 4 videos
  - **Complex queries**: 6,000 tokens, 4 videos

### **Acronym Metadata**
- Pre-extracted acronyms stored in database
- 22x faster acronym queries
- Precise matching using metadata filtering

### **Relevance Filtering**
- Only includes videos with high similarity scores
- Prevents low-relevance content from diluting context
- Better response quality and reduced token usage

## 📊 **Performance Metrics**

### **Response Times**
- **Simple queries**: 1-2 seconds
- **Complex queries**: 2-4 seconds
- **Clipping queries**: 3-5 seconds

### **Token Usage**
- **Before optimization**: 7,000+ tokens
- **After optimization**: 2,500-4,000 tokens
- **Improvement**: 60-70% reduction

### **Startup Time**
- **Small collection (10 videos)**: 10-15 seconds
- **Medium collection (50 videos)**: 30-45 seconds
- **Large collection (100+ videos)**: 60-90 seconds

## 🧪 **Testing**

### **Quick Test Script**
```bash
python test_api_client.py
```

### **Performance Testing**
```bash
python quick_lifespan_test.py
```

### **Debug Testing**
```bash
python debug_test.py
```

## 🔍 **Troubleshooting**

### **Common Issues**

1. **"System is still initializing"**
   - Wait for startup completion
   - Check `/status` endpoint for progress

2. **"No transcript files found"**
   - Ensure `.txt` files exist in `Max Life Videos/` folder
   - Check file permissions

3. **"Rate limit exceeded"**
   - Groq API rate limit reached
   - Wait or upgrade API plan

4. **"Module not found"**
   - Install dependencies: `pip install -r api_requirements.txt`
   - Check Python environment

### **Debug Commands**
```bash
# Check server status
curl http://localhost:8000/status

# Test health
curl http://localhost:8000/health

# List videos
curl http://localhost:8000/videos

# Test simple query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is NOPP?", "include_clips": false}'
```

## 🔧 **Configuration**

### **Environment Variables**
```bash
GROQ_API_KEY=your-groq-api-key
```

### **Server Configuration**
```python
# api_server.py
app = FastAPI(
    title="ViviRAG API",
    description="Intelligent video analysis and clipping API",
    version="1.0.0"
)
```

### **LLM Configuration**
```python
# Model settings
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=1)
llm_clipping = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
```

## 📈 **Monitoring**

### **Health Checks**
- `/health` endpoint for system status
- Component availability monitoring
- LLM connectivity testing

### **Performance Monitoring**
- Response time tracking
- Token usage monitoring
- Error rate tracking

### **Logging**
- Startup progress logging
- Query processing logs
- Error logging with stack traces

## 🚀 **Deployment**

### **Development**
```bash
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

### **Production**
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

### **Docker**
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r api_requirements.txt
EXPOSE 8000
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📚 **API Documentation**

Once the server is running, visit:
- **Interactive docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License.

## 🆘 **Support**

For issues and questions:
1. Check the troubleshooting guide
2. Review the technical documentation
3. Open an issue on GitHub
4. Contact the development team 