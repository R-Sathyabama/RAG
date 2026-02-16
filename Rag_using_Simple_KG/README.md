# 🧠 Simple Knowledge Graph RAG with Neo4j

> Load a FAQ file → Extract entities & relations → Store in Neo4j → Ask questions with AI

---

## 🔄 How It Works

```
FAQ.txt → LLM extracts entities & relations → Stored in Neo4j → Ask questions → AI answers!
```

---

## ⚡ Quick Setup (5 Minutes)

### 1. Create Neo4j Database (Free)

1. Go to [neo4j.com/cloud/aura-free](https://neo4j.com/cloud/aura-free/)
2. Sign up → Create **AuraDB Free** instance
3. Save your **URI**, **Username**, **Password**

### 2. Create Project

```bash
mkdir kg-rag && cd kg-rag
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install openai neo4j python-dotenv
```

### 4. Create `.env` File

```env
OPENAI_API_KEY=sk-your-key-here
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
```

### 5. Run It!

```bash
python app.py
```

---

## 🖼️ See Your Graph

Open Neo4j Browser and run:

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m
```

You'll see something like:

```
(AI) ←──SUBSET_OF── (Machine Learning) ←──SUBSET_OF── (Deep Learning)
 │                          │
 CREATED_BY              USED_IN
 │                          │
 ▼                          ▼
(Alan Turing)           (TensorFlow)──CREATED_BY──▶(Google)
```

---

## 🧩 How Each Step Works

| Step | What It Does | One-Line Explanation |
|------|-------------|---------------------|
| **Load** | Read `faq.txt` | Split FAQ into chunks by Q&A pairs |
| **Extract** | LLM reads each chunk | Returns entities (nouns) + relations (connections) |
| **Store** | Push to Neo4j | Creates nodes + edges in the graph database |
| **Query** | User asks question | Find matching graph nodes → send context to LLM → get answer |

### What is RAG?

```
Without RAG:  Question → LLM → Answer (guesses from training data)
With RAG:     Question → Search Database → LLM + Context → Accurate Answer ✅
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` | Check Neo4j is running + correct URI in `.env` |
| `AuthError` | Double-check Neo4j password in `.env` |
| `openai.AuthenticationError` | Check your OpenAI API key |
| Empty results | Make sure you ran the build phase first |

---

## 🚀 Want to Level Up?

- **Replace `faq.txt`** with your own company docs or PDFs
- **Change `gpt-4o-mini`** to `gpt-4o` for better extraction
- **Add a Streamlit UI** for a web interface
- **Combine with Vector RAG** for hybrid search

---

<p align="center">Built with ❤️ by <strong>Sathyabama</strong> 🎬</p>
