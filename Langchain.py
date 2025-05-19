import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import retrieval_qa
from langchain_community.chat_models import ChatOllama

# Configurations
PDF_PATH = "CV_for_Ashley.pdf"
INDEX_PATH = "vectorstore_index"
MODEL_NAME = "llama3"

# Initialize embeddings
embedding = HuggingFaceEmbeddings()

# Load or create vectorstore
if os.path.exists(INDEX_PATH):
    print("Loading existing vectorstore...")
    vectorstore = FAISS.load_local(INDEX_PATH, embedding, allow_dangerous_deserialization=True)
else:
    print("Creating new vectorstore...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)
    vectorstore = FAISS.from_documents(docs, embedding)
    vectorstore.save_local(INDEX_PATH)

# Setup RAG chain
qa = retrieval_qa.from_chain_type(
    llm=ChatOllama(model=MODEL_NAME),
    retriever=vectorstore.as_retriever(),
)

# Run query
response = qa.run("What is this paper about?")
print(response)

