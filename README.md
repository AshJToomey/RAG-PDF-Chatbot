# RAG PDF Chatbot with LangChain and Ollama

This project demonstrates a lightweight Retrieval-Augmented Generation (RAG) chatbot built using LangChain. It loads a PDF document, splits the content, embeds it using HuggingFace embeddings, and uses FAISS as the vector store. Responses are generated using a locally running LLM via Ollama (e.g., `llama3`).

---

## 🔧 Tech Stack

- **LangChain** — RAG orchestration
- **HuggingFace Embeddings** — For semantic search (no API key required)
- **FAISS** — Vector store for efficient retrieval
- **Ollama** — Local LLM interface (e.g. `llama3`)
- **PDF Loader** — Reads and chunks documents for ingestion

---

## 🚀 How It Works

1. Loads and splits a PDF file (`CV_for_Ashley.pdf`)
2. Generates document embeddings via HuggingFace
3. Stores embeddings in FAISS (in-memory)
4. Runs a LangChain `retrieval_qa` chain using an Ollama-hosted LLM
5. Accepts user queries and returns LLM-generated responses grounded in the PDF

---

## 📁 Project Structure

```
.
├── main.py                  # Main script
├── CV_for_Ashley.pdf       # Sample PDF for demo
├── requirements.txt        # Python dependencies
└── README.md               # Project overview (this file)
```

---

## 🧪 Usage

### 1. Start Ollama
Install [Ollama](https://ollama.com/) and run:
```bash
ollama run llama3
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
python main.py
```

### 4. Sample Output
```bash
> What is this paper about?
This document describes Ashley's qualifications, experience, and education...
```

---

## 📦 Requirements

- Python 3.8+
- Ollama installed locally
- PDF file (e.g. `CV_for_Ashley.pdf`)

---

## 📝 Notes
- The vectorstore is created in-memory each time the script runs. For persistence, you can use:
  ```python
  vectorstore.save_local("vectorstore_index")
  ```

---

## 📄 License
MIT — feel free to use and modify!
