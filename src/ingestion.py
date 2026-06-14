import os
from datasets import load_dataset
# Updated to use standalone langchain-chroma package instead of deprecated community version
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# Updated to the correct module path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def run_ingestion(data_path="mining_truck_service_manual.jsonl", persist_dir="chroma_db"):
    print("🚀 Loading data from JSONL file...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Please place your JSONL file in the root folder.")
        
    # Load dataset
    data = load_dataset('json', data_files=data_path, split='train')

    # Convert to LangChain Document objects
    documents = [
        Document(page_content=item['text'], metadata={'source': item.get('source', 'Unknown')}) 
        for item in data
    ]
    print(f"✅ Loaded {len(documents)} documents.")

    # Split Documents into Chunks using the updated package
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Split documents into {len(chunks)} chunks.")

    # Initialize Embedding Model
    print("📥 Initializing Hugging Face embedding model (all-MiniLM-L6-v2)...")
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Create and Persist the Vector Database
    print("📦 Creating and persisting the vector database...")
    db = Chroma.from_documents(
        chunks,
        embedding_function,
        persist_directory=persist_dir
    )
    print(f"🎉 Vector database created successfully inside '{persist_dir}'!")
    print(f"Total vectors indexed: {db._collection.count()}")

if __name__ == "__main__":
    run_ingestion(data_path="mining_truck_service_manual.jsonl", persist_dir="chroma_db")