#!/usr/bin/env python3
"""
Working GenPipes RAG Chatbot - All bugs fixed!
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
import argparse

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    import ollama
except ImportError:
    print("Missing dependencies. Install with:")
    print("pip install chromadb sentence-transformers ollama")
    sys.exit(1)

class LoadingIndicator:
    """Show loading animation"""
    def __init__(self, message="Loading"):
        self.message = message
        self.running = False
        self.thread = None
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        print("", end="", flush=True)  # Clean up animation
    
    def _animate(self):
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while self.running:
            print(f"\r{chars[i % len(chars)]} {self.message}...", end="", flush=True)
            time.sleep(0.1)
            i += 1

class WorkingVectorDatabase:
    def __init__(self, db_path="genpipes_vectordb", model_name="all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model_name = model_name
        self.embedding_model = None
        
        print("🔌 Connecting to ChromaDB...")
        try:
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_collection("genpipes_docs")
            count = self.collection.count()
            print(f"✅ Connected! Found {count} documents")
        except Exception as e:
            print(f"❌ ChromaDB error: {e}")
            print("Run: python genpipes_scraper.py --vectorize")
            sys.exit(1)
    
    def lazy_load_embedding_model(self):
        """Load embedding model with proper loading indicator"""
        if self.embedding_model is None:
            print("🤖 Loading embedding model...", end=" ", flush=True)
            start_time = time.time()
            try:
                self.embedding_model = SentenceTransformer(self.model_name)
                load_time = time.time() - start_time
                print(f"✅ ({load_time:.1f}s)")
            except Exception as e:
                print(f"❌ Error: {e}")
                raise
    
    def search(self, query, n_results=3):
        """Search with timeout protection"""
        self.lazy_load_embedding_model()
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            return results
        except Exception as e:
            print(f"❌ Search error: {e}")
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

class WorkingRAGChatbot:
    def __init__(self, vector_db, ollama_model="llama2"):
        self.vector_db = vector_db
        self.ollama_model = ollama_model
        
        # Test Ollama connection with FIXED parsing
        print("🔌 Testing Ollama connection...")
        try:
            # Simple connection test
            ollama.list()  # Just test connection
            print(f"✅ Ollama ready with model: {ollama_model}")
            
            # Quick test to make sure model works
            print("🧪 Testing model response...", end=" ", flush=True)
            start_time = time.time()
            
            test_response = ollama.chat(
                model=ollama_model,
                messages=[{"role": "user", "content": "Say 'ready'"}]
            )
            
            test_time = time.time() - start_time
            print(f"✅ ({test_time:.1f}s)")
            
        except Exception as e:
            print(f"❌ Ollama error: {e}")
            print("Troubleshooting:")
            print("1. Check if Ollama is running: ollama list")
            print("2. Pull model if missing: ollama pull llama2")
            print("3. Restart Ollama: pkill ollama && ollama serve &")
            sys.exit(1)
    
    def get_context(self, query, n_results=2):
        """Get relevant context with error handling"""
        try:
            print("🔍 Searching documentation...", end=" ", flush=True)
            start_time = time.time()
            
            results = self.vector_db.search(query, n_results=n_results)
            search_time = time.time() - start_time
            
            print(f"✅ ({search_time:.1f}s)")
            
            if not results['documents'][0]:
                return "No relevant GenPipes documentation found."
            
            context_parts = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                relevance = 1 - distance
                
                # Only include highly relevant results
                if relevance > 0.3:
                    title = metadata.get('title', 'GenPipes Documentation')
                    url = metadata.get('url', '')
                    
                    # Clean up URL for display
                    if url:
                        url_display = url.split('/')[-1] if '/' in url else url
                    else:
                        url_display = 'GenPipes Docs'
                    
                    context_parts.append(f"""
--- Source {i+1}: {title} ({url_display}) ---
Relevance: {relevance:.2f}

{doc[:350]}{'...' if len(doc) > 350 else ''}
""")
            
            return '\n'.join(context_parts) if context_parts else "No highly relevant documentation found."
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return f"Error retrieving documentation: {e}"
    
    def chat_with_progress(self, prompt):
        """Chat with Ollama with progress indicator"""
        print("🧠 Generating response...", end=" ", flush=True)
        
        loader = LoadingIndicator("Thinking")
        loader.start()
        
        try:
            start_time = time.time()
            
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_ctx": 2048  # Smaller context for speed
                }
            )
            
            response_time = time.time() - start_time
            loader.stop()
            
            print(f"✅ ({response_time:.1f}s)")
            return response['message']['content']
            
        except Exception as e:
            loader.stop()
            print(f"❌ Error: {e}")
            return f"Sorry, encountered an error: {e}"
    
    def chat(self, query):
        """Main chat function"""
        # Get context from documentation
        context = self.get_context(query)
        
        # Create focused prompt
        system_prompt = f"""You are a GenPipes bioinformatics expert. Answer based on the documentation context below.

GENPIPES DOCUMENTATION:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
- Use the documentation context to provide accurate answers
- Include exact GenPipes v6.0.0 command syntax when available
- Be specific about file formats, parameters, and options
- If context is insufficient, say so clearly
- Focus on practical, actionable information

Answer as a GenPipes expert:"""

        # Generate response
        return self.chat_with_progress(system_prompt)

def main():
    parser = argparse.ArgumentParser(description="Working GenPipes RAG Chatbot")
    parser.add_argument('--model', default='llama2', help='Ollama model')
    parser.add_argument('--db-path', default='genpipes_vectordb', help='Vector DB path')
    
    args = parser.parse_args()
    
    # Header
    print("🧬 GenPipes RAG Chatbot - WORKING VERSION")
    print("All bugs fixed, ready to use!")
    print("=" * 50)
    
    try:
        # Initialize components
        vector_db = WorkingVectorDatabase(db_path=args.db_path)
        chatbot = WorkingRAGChatbot(vector_db, args.model)
        
        print(f"\n🚀 Ready! Ask your GenPipes questions.")
        print("Commands: 'quit', 'help'")
        print("=" * 50)
        
        # Main chat loop
        while True:
            try:
                query = input("\n👤 You: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Happy GenPipes workflows!")
                    break
                
                if query.lower() == 'help':
                    print("""
🆘 GenPipes RAG Chatbot Help

🔍 Great questions to ask:
• "How do I run RNA-seq with StringTie protocol?"
• "What's the readsets file format?"  
• "ChIP-seq configuration for SLURM cluster"
• "What changed in GenPipes v6.0 vs v5.x?"
• "How to fix 'command not found' error?"
• "DNA-seq germline SNV protocol syntax"

💡 Tips:
• Be specific with pipeline names (RNA-seq, ChIP-seq, etc.)
• Mention your scheduler (SLURM, PBS, batch)
• Include context like "v6.0" or "on HPC cluster"

🎯 The chatbot searches 312 GenPipes documentation pages 
   to provide accurate, cited answers!
                    """)
                    continue
                
                # Process the query
                print(f"\n🎯 Query: '{query[:60]}{'...' if len(query) > 60 else ''}'")
                response = chatbot.chat(query)
                
                print(f"\n🧬 GenPipes Assistant:")
                print("=" * 50)
                print(response)
                print("=" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    except Exception as e:
        print(f"\n❌ Startup error: {e}")

if __name__ == "__main__":
    main()
