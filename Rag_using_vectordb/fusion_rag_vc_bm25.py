import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import BM25Retriever
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate

# -------------------------------------------------
# Streamlit Config
# -------------------------------------------------
st.set_page_config(page_title="Hybrid Offline RAG", layout="centered")
st.title("🔥 Hybrid Offline RAG (BM25 + Dense) – Phi3 Mini")

debug_mode = st.checkbox("🔍 Show internal reasoning")

mode = st.radio(
    "Choose data source",
    ["PDF Document", "Excel Document"]
)

# -------------------------------------------------
# Models
# -------------------------------------------------

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = get_embeddings()

llm = ChatOllama(
    model="phi3:mini",
    temperature=0
)

# -------------------------------------------------
# Prompt
# -------------------------------------------------

QA_PROMPT_TEMPLATE = """You must answer strictly from the provided context.
If the answer is not present, say "I don't know."

Context:
{context}

Question:
{question}

Answer:"""

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# -------------------------------------------------
# Query Classifier
# -------------------------------------------------

def classify_query(query: str) -> str:
    query_lower = query.lower()

    if any(word in query_lower for word in ['summarize', 'summary', 'overview']):
        return "summary"

    if any(word in query_lower for word in ['total', 'sum', 'calculate', 'count']):
        return "calculation"

    if query_lower in ['hello', 'hi', 'thanks']:
        return "no_rag"

    return "rag"

# -------------------------------------------------
# Hybrid Retriever
# -------------------------------------------------

def build_hybrid_retriever(text_chunks, vectordb):
    docs = [Document(page_content=chunk) for chunk in text_chunks]

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 4

    dense_retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    return bm25_retriever, dense_retriever


def hybrid_search(query, bm25_retriever, dense_retriever):
    bm25_docs = bm25_retriever.get_relevant_documents(query)
    dense_docs = dense_retriever.get_relevant_documents(query)

    combined = {doc.page_content: doc for doc in bm25_docs + dense_docs}

    return list(combined.values())

# -------------------------------------------------
# Adaptive Answer
# -------------------------------------------------

def adaptive_answer(query, vectordb, text_chunks):
    query_type = classify_query(query)

    if debug_mode:
        st.info(f"Detected Query Type: {query_type}")

    if query_type == "no_rag":
        return llm.invoke(query).content

    bm25_retriever, dense_retriever = build_hybrid_retriever(text_chunks, vectordb)
    docs = hybrid_search(query, bm25_retriever, dense_retriever)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = QA_PROMPT.format(context=context, question=query)

    response = llm.invoke(prompt)

    if debug_mode:
        st.subheader("Retrieved Documents")
        for i, doc in enumerate(docs, 1):
            with st.expander(f"Doc {i}"):
                st.write(doc.page_content)

    return response.content

# =================================================
# PDF MODE
# =================================================

if mode == "PDF Document":
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_pdf:
        raw_text = ""
        reader = PdfReader(uploaded_pdf)

        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text

        if not raw_text.strip():
            st.error("No readable text found.")
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_text(raw_text)

            vectordb = Chroma.from_texts(
                chunks,
                embedding=embeddings,
                persist_directory="chroma_pdf"
            )

            st.success(f"PDF processed ({len(chunks)} chunks)")

            query = st.text_input("Ask question about PDF:")

            if query:
                answer = adaptive_answer(query, vectordb, chunks)
                st.subheader("Answer")
                st.write(answer)

# =================================================
# EXCEL MODE
# =================================================

if mode == "Excel Document":
    uploaded_excel = st.file_uploader("Upload Excel", type=["xlsx", "xls"])

    if uploaded_excel:
        df = pd.read_excel(uploaded_excel)

        excel_text = []
        headers = "Columns: " + ", ".join(df.columns.tolist())
        excel_text.append(headers)

        for idx, row in df.iterrows():
            row_text = f"Row {idx+1}: " + " | ".join(
                f"{col}={row[col]}" for col in df.columns
            )
            excel_text.append(row_text)

        vectordb = Chroma.from_texts(
            excel_text,
            embedding=embeddings,
            persist_directory="chroma_excel"
        )

        st.success(f"Excel processed ({len(df)} rows)")

        query = st.text_input("Ask question about Excel:")

        if query:
            answer = adaptive_answer(query, vectordb, excel_text)
            st.subheader("Answer")
            st.write(answer)
