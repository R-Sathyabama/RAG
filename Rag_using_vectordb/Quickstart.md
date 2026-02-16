# 🚀 Quick Start Guide - Offline Adaptive RAG

## 5-Minute Setup

### Step 1: Install Ollama (2 minutes)

**macOS / Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from https://ollama.com/download/windows

**Verify Installation:**
```bash
ollama --version
```

### Step 2: Download Phi3 Mini Model (1 minute)

```bash
ollama pull phi3:mini
```

This downloads a ~2.3GB model. Wait for completion.

**Verify:**
```bash
ollama list
# Should show phi3:mini
```

### Step 3: Install Python Dependencies (1 minute)

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Run the Application (30 seconds)

**For Standard RAG:**
```bash
streamlit run file_one.py
```

**For Hybrid RAG (recommended):**
```bash
streamlit run file_two.py
```

Your browser will open at `http://localhost:8501`

---

## First Query Example

### For PDF Documents:

1. Upload a PDF file
2. Wait for processing (you'll see: "PDF processed (X chunks)")
3. Ask questions like:
   - "What is this document about?"
   - "Summarize the main points"
   - "What does section 2 discuss?"

### For Excel Files:

1. Upload an Excel file
2. Wait for processing
3. Ask questions like:
   - "What columns are available?"
   - "What is the total of column Revenue?"
   - "Show me information from row 5"

---

## Troubleshooting

### "Cannot connect to Ollama"
```bash
# Start Ollama service
ollama serve
```

### "Model not found"
```bash
# Pull the model again
ollama pull phi3:mini
```

### "Memory error"
- Close other applications
- Try smaller documents first
- Reduce chunk size in code if needed

---

## What's Next?

1. ✅ Try both variants (file_one.py and file_two.py)
2. ✅ Enable debug mode to see retrieval process
3. ✅ Experiment with different document types
4. ✅ Compare Standard vs Hybrid RAG accuracy
5. ✅ Star the repo if you find it useful!

---

## Advanced Usage

### Custom Chunk Size
Edit in the Python file:
```python
chunk_size=500,      # Smaller chunks
chunk_overlap=100    # Less overlap
```

### Change Retrieval Count
```python
search_kwargs={"k": 6}  # Retrieve 6 instead of 4
```

### Use Different Model
```bash
# Download another model
ollama pull llama2

# Update in code
llm = ChatOllama(model="llama2", temperature=0)
```

---

## Performance Tips

1. **First run is slower** - Models load into memory
2. **Keep Ollama running** - Don't restart between queries
3. **Use SSD** - Faster for vector database
4. **Close browser tabs** - Free up RAM
5. **Process large files in batches** - Split PDFs if too large

---

## Community & Support

- **Report Issues:** GitHub Issues
- **Feature Requests:** GitHub Discussions
- **Questions:** Open an issue with "Question" label
- **Contribute:** Fork, branch, PR!

---

**Happy RAG-ing! 🚀**