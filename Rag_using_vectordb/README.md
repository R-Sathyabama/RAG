# 🔥 Offline Adaptive RAG System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-Latest-yellow.svg)

A powerful, privacy-focused **Retrieval-Augmented Generation (RAG)** system that runs completely offline using local LLMs and embeddings. Perfect for sensitive documents, air-gapped environments, or anyone who values data privacy.

## 🌟 Key Features

- **100% Offline** - No API keys, no cloud, no data leakage
- **Dual Architecture** - Choose between Standard RAG or Hybrid (BM25 + Dense) retrieval
- **Multi-Format Support** - PDF and Excel documents
- **Smart Query Classification** - Automatically detects query type for optimal responses
- **Local LLM** - Powered by Phi3 Mini via Ollama
- **Privacy First** - All processing happens on your machine

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [System Variants](#system-variants)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🏗️ Architecture Overview

```
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Query Classifier    │  ← Detects: Summary/Calculation/RAG/General
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Retrieval Strategy  │
├─────────────────────┤
│ • BM25 (Keyword)    │  ← Hybrid Mode Only
│ • Dense (Semantic)  │
│ • Hybrid Fusion     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Context Formation   │  ← Top-K relevant chunks
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ LLM Generation      │  ← Phi3 Mini (Ollama)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Structured Answer   │
└─────────────────────┘
```

## 🔧 System Variants

### Variant 1: Standard Adaptive RAG
**File:** `adaptive_rag_vc.py`

**Best For:**
- General Q&A over documents
- Simple deployments
- When semantic search is sufficient

**Features:**
- Dense vector retrieval using sentence transformers
- Query type classification
- Optimized prompts for different query types
- Debug mode with source document tracking

### Variant 2: Hybrid RAG (BM25 + Dense)
**File:** `fusion_rag_vc_bm25.py`

**Best For:**
- Complex queries requiring both keyword and semantic matching
- Technical documents with specific terminology
- Maximum retrieval accuracy

**Features:**
- BM25 (keyword-based) retrieval
- Dense vector (semantic) retrieval
- Automatic fusion of both approaches
- Deduplication of results

## 📦 Installation

### Prerequisites

1. **Python 3.8+**
```bash
python --version
```

2. **Ollama with Phi3 Mini**
```bash
# Install Ollama (https://ollama.ai)
curl -fsSL https://ollama.com/install.sh | sh

# Pull Phi3 Mini model
ollama pull phi3:mini
```

### Step-by-Step Setup

```bash
# 1. Clone or download the repository
git clone <your-repo-url>
cd offline-adaptive-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install streamlit pandas PyPDF2 openpyxl
pip install langchain langchain-community
pip install chromadb sentence-transformers
pip install rank-bm25

or

pip install -r requirements.txt

# 4. Verify Ollama is running
ollama list  # Should show phi3:mini
```

## 🚀 Quick Start

### Running Standard RAG (Variant 1)
```bash
streamlit run file_one.py
```

### Running Hybrid RAG (Variant 2)
```bash
streamlit run file_two.py
```

The app will open in your browser at `http://localhost:8501`

## 💡 Usage Examples

### PDF Document Analysis

1. **Upload Document**
   - Select "PDF Document" mode
   - Upload your PDF file
   - Wait for processing (you'll see chunk count)

2. **Ask Questions**
   ```
   Example queries:
   - "What is this document about?"
   - "Summarize the main findings"
   - "What does section 3 say about methodology?"
   - "List all recommendations"
   ```

### Excel Data Analysis

1. **Upload Spreadsheet**
   - Select "Excel Document" mode
   - Upload .xlsx or .xls file
   - System processes all rows

2. **Query Your Data**
   ```
   Example queries:
   - "What columns are in this dataset?"
   - "What is the total revenue in Q2?"
   - "Show me information about row 15"
   - "Summarize the sales trends"
   ```

## 🔬 Technical Details

### Query Classification System

The system automatically routes queries based on intent:

| Query Type | Triggers | Behavior |
|------------|----------|----------|
| **Summary** | "summarize", "overview", "main points" | Retrieves 6 chunks, uses summary prompt |
| **Calculation** | "total", "sum", "calculate", "count" | Retrieves 5 chunks, uses QA prompt |
| **RAG** | Standard questions | Retrieves 4 chunks, standard context |
| **No RAG** | "hello", "thanks" | Direct LLM response without retrieval |

### Embedding Model

**Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions:** 384
- **Max Sequence Length:** 256 tokens
- **Speed:** ~1000 sentences/second on CPU
- **Size:** ~80MB

### LLM Configuration

**Model:** Phi3 Mini (via Ollama)
- **Parameters:** 3.8B
- **Context Window:** 4K tokens
- **Temperature:** 0 (deterministic)
- **Quantization:** 4-bit (default Ollama)

### Chunking Strategy

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,        # ~750 tokens
    chunk_overlap=200,      # 20% overlap
    separators=[
        "\n\n",  # Paragraphs
        "\n",    # Lines
        ".",     # Sentences
        " ",     # Words
        ""       # Characters
    ]
)
```

### Hybrid Retrieval (Variant 2 Only)

**BM25 Scoring:**
```
score(D,Q) = Σ IDF(qi) × (f(qi,D) × (k1 + 1)) / (f(qi,D) + k1 × (1 - b + b × |D|/avgdl))
```

**Dense Retrieval:**
- Cosine similarity in 384-dimensional space
- HNSW indexing via ChromaDB

**Fusion Strategy:**
- Union of top-K from both retrievers
- Deduplication by content hash
- Preserves document objects with metadata

## 🐛 Troubleshooting

### Common Issues

**1. "Cannot connect to Ollama"**
```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve

# Verify model is downloaded
ollama pull phi3:mini
```

**2. "No module named 'chromadb'"**
```bash
pip install chromadb sentence-transformers
```

**3. "PDF has no readable text"**
- PDF might be image-based (scanned)
- Try OCR preprocessing with Tesseract
- Or use a different PDF

**4. Memory Issues**
```bash
# Reduce chunk size in code
chunk_size=500  # Instead of 1000

# Or limit retrieval
search_kwargs={"k": 2}  # Instead of 4
```

**5. Slow Processing**
- Embedding generation is CPU-intensive
- Consider using GPU if available:
  ```python
  embeddings = HuggingFaceEmbeddings(
      model_name="...",
      model_kwargs={'device': 'cuda'}
  )
  ```

## 📊 Performance Benchmarks

| Task | Standard RAG | Hybrid RAG |
|------|--------------|------------|
| PDF Processing (100 pages) | ~45s | ~60s |
| Excel Processing (1000 rows) | ~30s | ~40s |
| Query Response | ~2-5s | ~3-6s |
| Memory Usage | ~1.5GB | ~2GB |

*Tested on: Intel i7, 16GB RAM, No GPU*

## 🔐 Privacy & Security

- **Zero Network Calls:** Everything runs locally
- **No Telemetry:** No usage data collected
- **No API Keys:** No credentials needed
- **Data Residency:** Files never leave your machine
- **Temporary Storage:** ChromaDB persists to disk but can be cleared

**Clear Stored Data:**
```bash
rm -rf chroma_pdf/
rm -rf chroma_excel/
```

## 📈 Roadmap

- [ ] Support for .docx files
- [ ] Multi-language document support
- [ ] Conversation memory across queries
- [ ] Export Q&A pairs
- [ ] GPU acceleration option
- [ ] Larger LLM support (Llama, Mistral)
- [ ] Citation tracking (page numbers)
- [ ] Batch query processing

## 🤝 Contributing

Contributions welcome! Areas of interest:

1. **New Retrievers:** Implement alternative retrieval strategies
2. **LLM Support:** Add support for other local LLMs
3. **File Formats:** Word, PowerPoint, CSV parsers
4. **UI/UX:** Improve Streamlit interface
5. **Performance:** Optimization and caching
6. **Documentation:** Tutorials and examples

### Development Setup

```bash
# Fork and clone the repo
git clone <your-fork-url>
cd offline-adaptive-rag

# Create feature branch
git checkout -b feature/your-feature

# Install dev dependencies
pip install -r requirements-dev.txt  # If available

# Make changes and test
streamlit run file_one.py

# Submit PR
git push origin feature/your-feature
```

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **LangChain** - RAG framework
- **Ollama** - Local LLM serving
- **Sentence Transformers** - Embedding models
- **Streamlit** - Web interface
- **ChromaDB** - Vector database

## 📞 Support & Community

- **Issues:** GitHub Issues
- **Discussions:** GitHub Discussions

## ⭐ Star History

If this project helped you, please consider giving it a star!

---

**Made with ❤️ for the open-source and privacy-conscious community**

*Last Updated: February 2026*