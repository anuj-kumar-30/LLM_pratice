# langchain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
## FAISS vector storing
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import os
from dotenv import load_dotenv
load_dotenv()

print("🚀 Initializing embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("✅ Embedding model loaded successfully!")

print("\n📁 Loading PDF documents...")
docs_loader = DirectoryLoader(
    path=".",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader
)
docs = docs_loader.load()
print(f"✅ Loaded {len(docs)} PDF documents")

print("\n✂️ Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)
print(f"✅ Created {len(chunks)} text chunks")

print("\n🔧 Creating FAISS vector store with batch processing...")

# Process chunks in batches of 50
batch_size = 50
total_chunks = len(chunks)
vector_store = None

for i in range(0, total_chunks, batch_size):
    batch_end = min(i + batch_size, total_chunks)
    current_batch = chunks[i:batch_end]
    
    # Calculate progress
    progress = ((batch_end) / total_chunks) * 100
    batch_num = (i // batch_size) + 1
    total_batches = (total_chunks + batch_size - 1) // batch_size
    
    print(f"📦 Processing batch {batch_num}/{total_batches} (chunks {i+1}-{batch_end}) - {progress:.1f}% complete")
    
    if vector_store is None:
        # Create initial vector store with first batch
        vector_store = FAISS.from_documents(
            documents=current_batch,
            embedding=embedding_model
        )
        print(f"   ✅ Initial vector store created with {len(current_batch)} chunks")
    else:
        # Add subsequent batches to existing vector store
        vector_store.add_documents(current_batch)
        print(f"   ✅ Added {len(current_batch)} chunks to vector store")
    
    # Show progress bar
    filled = int(progress // 2)  # Scale to 50 chars
    bar = "█" * filled + "░" * (50 - filled)
    print(f"   [{bar}] {progress:.1f}%")
    print()

print("✅ Vector store created successfully with all chunks!")

# Vector store information
print(f"\n📊 Vector Store Details:")
print(f"   - Total vectors: {vector_store.index.ntotal}")
print(f"   - Vector dimension: {vector_store.index.d}")
print(f"   - Index type: {type(vector_store.index).__name__}")

# Test similarity search
vector_store.save_local("faiss_index")
