#!/bin/bash
# Test the fixed GenPipes RAG chatbot

echo "🧪 Testing Fixed GenPipes RAG Chatbot"
echo "======================================"

cd ~/genpipes-rag-chatbot  # Adjust path as needed
source venv/bin/activate

echo "1️⃣ Testing Ollama (should respond quickly now)..."
timeout 15s ollama run llama2 "Say 'Ollama working'" 2>/dev/null && echo "✅ Ollama OK" || echo "❌ Ollama timeout"

echo ""
echo "2️⃣ Testing ChromaDB..."
python -c "
import chromadb
try:
    client = chromadb.PersistentClient(path='genpipes_vectordb')
    collection = client.get_collection('genpipes_docs')
    print(f'✅ ChromaDB OK - {collection.count()} documents')
except Exception as e:
    print(f'❌ ChromaDB error: {e}')
"

echo ""
echo "3️⃣ Quick embedding test..."
python -c "
from sentence_transformers import SentenceTransformer
import time
start = time.time()
model = SentenceTransformer('all-MiniLM-L6-v2')
elapsed = time.time() - start
print(f'✅ Embedding model loaded in {elapsed:.1f}s')
"

echo ""
echo "4️⃣ Testing the fixed chatbot..."
echo "Starting chatbot - it should load much faster now!"
echo ""
echo "Try asking: 'How do I run RNA-seq pipeline?'"
echo "Expected: Should respond in 10-30 seconds (not hang)"
echo ""
echo "If it works, you'll see:"
echo "- ✅ Connected! Found XXX documents"
echo "- ✅ Ollama ready with model: llama2"  
echo "- 🔍 Searching documentation... ✅"
echo "- 🧠 Generating response... ✅"
echo ""
echo "🚀 Starting fixed chatbot now..."
echo "==============================="

python fixed_rag_chatbot.py --model llama2
