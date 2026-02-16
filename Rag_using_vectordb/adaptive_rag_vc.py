import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate

# -------------------------------------------------
# Setup
# -------------------------------------------------
st.set_page_config(page_title="Offline Adaptive RAG", layout="centered")
st.title("🧠 Offline Adaptive RAG – Phi3 Mini")

debug_mode = st.checkbox("🔍 Show internal reasoning (debug mode)")

mode = st.radio(
    "Choose data source",
    ["PDF Document", "Excel Document"]
)

# -------------------------------------------------
# Models (Fully Offline)
# -------------------------------------------------

# Local embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = get_embeddings()

# Local LLM (Ollama)
llm = ChatOllama(
    model="phi3:mini",
    temperature=0
)

# -------------------------------------------------
# Improved Custom Prompts
# -------------------------------------------------
QA_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Provide a clear, direct answer in simple human language.

Context:
{context}

Question: {question}

Helpful Answer:"""

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

SUMMARY_PROMPT_TEMPLATE = """Based on the following context, provide a clear and concise summary.
Focus on the main points and key information.

Context:
{context}

Question: {question}

Summary:"""

SUMMARY_PROMPT = PromptTemplate(
    template=SUMMARY_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# -------------------------------------------------
# Query Classification (Improved)
# -------------------------------------------------
def classify_query(query: str) -> str:
    """Classify the query into different types"""
    query_lower = query.lower()
    
    # Simple keyword-based classification
    if any(word in query_lower for word in ['summarize', 'summary', 'overview', 'what is this about', 'main points']):
        return "summary"
    
    if any(word in query_lower for word in ['calculate', 'total', 'sum', 'how much', 'how many', 'add', 'count']):
        return "calculation"
    
    # Check if query is too general or doesn't need RAG
    general_queries = ['hello', 'hi', 'thanks', 'thank you', 'bye']
    if query_lower in general_queries:
        return "no_rag"
    
    # Default to RAG for document-based questions
    return "rag"

# -------------------------------------------------
# Adaptive Strategy (FIXED)
# -------------------------------------------------
def adaptive_answer(query, vectordb):
    query_type = classify_query(query)

    if debug_mode:
        st.info(f"🧠 Query type detected: {query_type}")

    # For general queries that don't need document context
    if query_type == "no_rag":
        response = llm.invoke(query)
        return response.content

    # For summary requests
    elif query_type == "summary":
        retriever = vectordb.as_retriever(search_kwargs={"k": 6})
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": SUMMARY_PROMPT}
        )
        result = qa.invoke({"query": query})
        return result['result']

    # For calculation/numerical questions
    elif query_type == "calculation":
        # First get context from document
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
        result = qa.invoke({"query": query})
        return result['result']

    # Default RAG for factual questions
    else:
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
        result = qa.invoke({"query": query})
        
        if debug_mode:
            # Show source documents
            st.subheader("📚 Source Documents Used:")
            docs = retriever.get_relevant_documents(query)
            for i, doc in enumerate(docs, 1):
                with st.expander(f"Source {i}"):
                    st.write(doc.page_content)
        
        return result['result']

# =================================================
# PDF MODE
# =================================================
if mode == "PDF Document":
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_pdf:
        with st.spinner("Processing PDF..."):
            raw_text = ""
            reader = PdfReader(uploaded_pdf)

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    raw_text += text

        if not raw_text.strip():
            st.error("❌ No readable text found")
        else:
            # Better chunking strategy
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )

            chunks = splitter.split_text(raw_text)

            # Create vector store
            with st.spinner("Creating vector database..."):
                vectordb = Chroma.from_texts(
                    chunks,
                    embedding=embeddings,
                    persist_directory="chroma_pdf"
                )

            st.success(f"✅ PDF processed ({len(chunks)} chunks)")

            # Question input
            query = st.text_input("Ask a question about the document:")

            if query:
                with st.spinner("Thinking..."):
                    answer = adaptive_answer(query, vectordb)

                st.subheader("📌 Answer")
                st.write(answer)
                
                # Show example questions
                with st.expander("💡 Example Questions"):
                    st.markdown("""
                    - What is this document about?
                    - Summarize the main points
                    """)

# =================================================
# EXCEL MODE
# =================================================
if mode == "Excel Document":
    uploaded_excel = st.file_uploader(
        "Upload Excel file",
        type=["xlsx", "xls"]
    )

    if uploaded_excel:
        df = pd.read_excel(uploaded_excel)
        st.success(f"✅ Excel loaded ({len(df)} rows, {len(df.columns)} columns)")

        # Better Excel text representation
        excel_text = []
        
        # Add column headers as context
        headers = "Columns: " + ", ".join(df.columns.tolist())
        excel_text.append(headers)
        
        # Add each row with better formatting
        for idx, row in df.iterrows():
            row_text = f"Row {idx + 1}: " + " | ".join(
                f"{col}={row[col]}" for col in df.columns
            )
            excel_text.append(row_text)

        # Create vector store
        with st.spinner("Processing Excel data..."):
            vectordb = Chroma.from_texts(
                excel_text,
                embedding=embeddings,
                persist_directory="chroma_excel"
            )

        st.success(f"✅ Excel processed ({len(df)} rows)")

        query = st.text_input("Ask your question about the data:")

        if query:
            with st.spinner("Analyzing..."):
                answer = adaptive_answer(query, vectordb)

            st.subheader("📌 Answer")
            st.write(answer)
            
            # Show example questions for Excel
            with st.expander("💡 Example Questions"):
                st.markdown(f"""
                - What columns are in this data?
                - Show me information about row 5
                - What is the value of [column_name] in row 3?
                - Summarize the data
                """)