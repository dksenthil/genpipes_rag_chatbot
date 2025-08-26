#!/usr/bin/env python3
"""
Debug GenPipes RAG Chatbot - Find and fix hanging issues
"""

import os
import sys
import time
import signal
from pathlib import Path

def test_ollama():
    """Test if Ollama is working"""
    print("🔍 Testing Ollama connection...")
    try:
        import ollama
        
        # Test basic connection
        models = ollama.list()
        print(f"✅ Ollama connected, {len(models['models'])} models available")
        
        # Test specific model
        model_name = "llama2"
        available_models = [m['name'] for m in models['models']]
        
        if model_name not in available_models:
            print(f"❌ Model '{model_name}' not found")
            print(f"Available models: {available_models}")
            return False
        
        print(f"✅ Model '{model_name}' is available")
        
        # Test simple chat with timeout
        print("🧪 Testing simple chat...")
        def timeout_handler(signum, frame):
            raise TimeoutError("Ollama chat timeout")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        try:
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": "Hello, respond with just 'Hi'"}]
            )
            signal.alarm(0)  # Cancel timeout
            print(f"✅ Ollama chat working: '{response['message']['content'][:50]}...'")
            return True
            
        except TimeoutError:
            print("❌ Ollama chat timeout - model may be slow or stuck")
            return False
        except Exception as e:
            signal.alarm(0)
            print(f"❌ Ollama chat error: {e}")
            return False
            
    except ImportError:
        print("❌ Ollama not installed: pip install ollama")
        return False
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return False

def test_vector_db():
    """Test vector database"""
    print("\n🔍 Testing ChromaDB...")
    try:
        import chromadb
        
        db_path = "genpipes_vectordb"
        if not Path(db_path).exists():
            print(f"❌ Vector database not found at {db_path}")
            return False
        
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection("genpipes_docs")
        count = collection.count()
        print(f"✅ ChromaDB connected, {count} documents")
        
        # Test search
        print("🧪 Testing vector search...")
        start_time = time.time()
        results = collection.query(
            query_texts=["GenPipes RNA-seq pipeline"],
            n_results=2
        )
        search_time = time.time() - start_time
        
        if results['documents'][0]:
            print(f"✅ Vector search working ({search_time:.2f}s)")
            return True
        else:
            print("❌ Vector search returned no results")
            return False
            
    except Exception as e:
        print(f"❌ ChromaDB error: {e}")
        return False

def test_embedding_model():
    """Test embedding model"""
    print("\n🔍 Testing embedding model...")
    try:
        from sentence_transformers import SentenceTransformer
        
        print("🤖 Loading all-MiniLM-L6-v2...")
        start_time = time.time()
        model = SentenceTransformer('all-MiniLM-L6-v2')
        load_time = time.time() - start_time
        print(f"✅ Model loaded ({load_time:.2f}s)")
        
        # Test encoding
        print("🧪 Testing text encoding...")
        start_time = time.time()
        embedding = model.encode(["test sentence"])
        encode_time = time.time() - start_time
        
        print(f"✅ Encoding working ({encode_time:.2f}s, shape: {embedding.shape})")
        return True
        
    except Exception as e:
        print(f"❌ Embedding model error: {e}")
        return False

def create_simple_chatbot():
    """Create a minimal working chatbot for testing"""
    print("\n🔧 Creating simple test chatbot...")
    
    simple_bot = """#!/usr/bin/env python3
# simple_test_bot.py - Minimal chatbot for debugging

import ollama
import chromadb
import time

def main():
    print("🧪 Simple GenPipes Test Bot")
    print("Testing each component...")
    
    # Test Ollama
    try:
        print("1. Testing Ollama...", end=" ")
        response = ollama.chat(
            model="llama2", 
            messages=[{"role": "user", "content": "Say 'working' only"}]
        )
        print(f"✅ {response['message']['content'][:20]}")
    except Exception as e:
        print(f"❌ {e}")
        return
    
    # Test ChromaDB
    try:
        print("2. Testing ChromaDB...", end=" ")
        client = chromadb.PersistentClient(path="genpipes_vectordb")
        collection = client.get_collection("genpipes_docs")
        count = collection.count()
        print(f"✅ {count} docs")
    except Exception as e:
        print(f"❌ {e}")
        return
    
    # Test search
    try:
        print("3. Testing search...", end=" ")
        results = collection.query(query_texts=["RNA-seq"], n_results=1)
        doc = results['documents'][0][0][:50] if results['documents'][0] else "No docs"
        print(f"✅ {doc}...")
    except Exception as e:
        print(f"❌ {e}")
        return
    
    print("\\n🚀 All tests passed! Starting simple chat...")
    
    while True:
        try:
            query = input("\\n👤 You: ").strip()
            if query.lower() in ['quit', 'exit']:
                break
            if not query:
                continue
            
            print("🔍 Searching...", end=" ", flush=True)
            results = collection.query(query_texts=[query], n_results=1)
            context = results['documents'][0][0][:200] if results['documents'][0] else ""
            print("✅", end=" ", flush=True)
            
            print("🧠 Thinking...", end=" ", flush=True)
            prompt = f"Context: {context}\\n\\nQuestion: {query}\\n\\nAnswer briefly:"
            
            response = ollama.chat(
                model="llama2",
                messages=[{"role": "user", "content": prompt}]
            )
            print("✅")
            
            print(f"🤖: {response['message']['content']}")
            
        except KeyboardInterrupt:
            print("\\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\\n❌ Error: {e}")

if __name__ == "__main__":
    main()
"""
    
    with open("simple_test_bot.py", "w") as f:
        f.write(simple_bot)
    
    print("✅ Created simple_test_bot.py")
    print("Run with: python simple_test_bot.py")

def main():
    print("🔧 GenPipes RAG Chatbot Debugger")
    print("=" * 50)
    
    # Run all tests
    ollama_ok = test_ollama()
    vector_ok = test_vector_db()
    embed_ok = test_embedding_model()
    
    print("\n📊 Test Results:")
    print(f"Ollama: {'✅' if ollama_ok else '❌'}")
    print(f"Vector DB: {'✅' if vector_ok else '❌'}")
    print(f"Embedding: {'✅' if embed_ok else '❌'}")
    
    if all([ollama_ok, vector_ok, embed_ok]):
        print("\n🎉 All components working!")
        print("The hanging issue might be due to:")
        print("1. Slow model inference (normal for first few queries)")
        print("2. Large context causing timeout")
        print("3. Resource contention")
        
        create_simple_chatbot()
        
    else:
        print("\n🔧 Fixes needed:")
        if not ollama_ok:
            print("- Start Ollama: ollama serve")
            print("- Check model: ollama pull llama2")
        if not vector_ok:
            print("- Rebuild vector DB: python genpipes_scraper.py --vectorize")
        if not embed_ok:
            print("- Fix dependencies: pip install sentence-transformers")
    
    print(f"\n💡 Quick fixes:")
    print("1. Try the simple test bot: python simple_test_bot.py")
    print("2. Restart Ollama: pkill ollama && ollama serve &")
    print("3. Use smaller model: --model mistral")
    print("4. Check system resources: top")

if __name__ == "__main__":
    main()
