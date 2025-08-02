# 📄 RAG-based Question Answering System for Policy Documents

This project demonstrates a complete **Retrieval-Augmented Generation (RAG)** pipeline for querying complex policy documents using semantic search and large language models (LLMs). It enables accurate, context-aware responses to natural language questions by combining document retrieval with generative AI.

---

## 🚀 Project Highlights

- 📥 **Document Ingestion**: Loads multiple `.txt` policy files using `TextLoader` from LangChain.
- ✂️ **Chunking**: Uses `RecursiveCharacterTextSplitter` to split documents into manageable overlapping chunks for better context preservation.
- 🧠 **Vector Store Creation**: Embeds chunks using `HuggingFaceEmbeddings` and stores them in a FAISS index for fast similarity search.
- 🔍 **Document Retrieval**: Retrieves relevant chunks using FAISS based on a user’s query.
- 🤖 **LLM Integration**: Combines retrieved context with a prompt and passes it through an LLM to generate structured answers.

---

## 🛠️ Technologies Used

- Python
- LangChain
- FAISS (Facebook AI Similarity Search)
- Hugging Face Transformers
- OpenAI (optional, for LLM-based components)

---

## 🧱 Key Pipeline Steps

1. **Document Loading**
   - Reads policy `.txt` files using `TextLoader`.

2. **Chunking**
   - Applies `RecursiveCharacterTextSplitter` with overlap to maintain context flow across chunks.

3. **Embeddings + Vector Store**
   - Generates embeddings using `HuggingFaceEmbeddings`.
   - Stores chunks in a FAISS vector store and saves to disk.

4. **Query Handling**
   - Accepts natural language questions.
   - Retrieves top matching chunks from the vector store.
   - Prepares a context-rich prompt.

5. **LLM Prompting**
   - Uses LangChain prompt templates with a structured output parser.
   - Passes prompt to an LLM to return a structured summary and category.

---

## 📦 Output Example

**Input Question:**  
`"Give introduction from Comprehensive AI Ethics Policy Document"`

**Returned Output:**
```json
{
  "summary": "This document outlines the company's ethical principles for developing and using AI.",
  "category": "AI Governance"
}

---

## 📁 Folder Structure
```
project_root/
├── RAG_final.ipynb          # Main pipeline notebook
├── data/
│   ├── policy1.txt
│   ├── policy2.txt
│   └── policy3.txt
└── vectorstore/
    └── faiss_index.faiss    # Saved vector index
```
---

✅ Ideal Use Cases

    AI policy compliance tools

    Document intelligence for enterprises

    Legal and compliance document summarization

    Context-aware chatbots with deep domain knowledge

---

## 🚀 Future Enhancements

    UI integration (e.g., Streamlit or Gradio)

    Long document support using memory and history

    Support for uploading PDFs via PyMuPDF or pdfplumber

    Switch from FAISS to Weaviate, Pinecone, or ChromaDB
