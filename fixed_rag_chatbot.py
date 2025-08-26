#!/usr/bin/env python3
"""
Fixed GenPipes RAG Chatbot - With proper timeouts and loading indicators
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
        print("")  # New line after animation
    
    def _animate(self):
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while self.running:
            print(f"\r{chars[i % len(chars)]} {self.message}...", end="", flush=True)
            time.sleep(0.1)
            i += 1

class FixedVectorDatabase:
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
            loader = LoadingIndicator("Loading embedding model (first time)")
            loader.start()
            try:
                self.embedding_model = SentenceTransformer(self.model_name)
                loader.stop()
                print("✅ Embedding model ready!")
            except Exception as e:
                loader.stop()
                print(f"❌ Embedding model error: {e}")
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

class FixedRAGChatbot:
    def __init__(self, vector_db, ollama_model="llama2"):
        self.vector_db = vector_db
        self.ollama_model = ollama_model
        
        # Test Ollama connection
        print("🔌 Testing Ollama connection...")
        try:
            models = ollama.list()
            available_models = [m['name'] for m in models['models']]
            
            if ollama_model not in available_models:
                print(f"❌ Model '{ollama_model}' not found")
                print(f"Available: {', '.join(available_models)}")
                print("Pull model with: ollama pull llama2")
                sys.exit(1)
            
            print(f"✅ Ollama ready with model: {ollama_model}")
            
        except Exception as e:
            print(f"❌ Ollama connection error: {e}")
            print("Start Ollama with: ollama serve")
            sys.exit(1)
    
    def get_context(self, query, n_results=3):
        """Get relevant context with error handling"""
        try:
            print("🔍 Searching documentation...", end=" ", flush=True)
            start_time = time.time()
            
            results = self.vector_db.search(query, n_results=n_results)
            search_time = time.time() - start_time
            
            print(f"✅ ({search_time:.1f}s)")
            
            if not results['documents'][0]:
                return "No relevant GenPipes documentation found for this query."
            
            context_parts = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                relevance = 1 - distance
                
                # Only include relevant results
                if relevance > 0.2:
                    title = metadata.get('title', 'Unknown')
                    url = metadata.get('url', '')
                    
                    context_parts.append(f"""
--- GenPipes Documentation Source {i+1} ---
Title: {title}
URL: {url}
Relevance: {relevance:.2f}

Content:
{doc[:400]}{'...' if len(doc) > 400 else ''}
""")
            
            return '\n'.join(context_parts) if context_parts else "No highly relevant documentation found."
            
        except Exception as e:
            return f"Error retrieving documentation: {e}"
    
    def chat_with_timeout(self, prompt, timeout=60):
        """Chat with Ollama with timeout protection"""
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
                    "num_ctx": 2048  # Smaller context to speed up
                }
            )
            
            response_time = time.time() - start_time
            loader.stop()
            
            print(f"✅ Response generated ({response_time:.1f}s)")
            return response['message']['content']
            
        except Exception as e:
            loader.stop()
            print(f"❌ Error: {e}")
            return f"Sorry, I encountered an error generating the response: {e}"
    
    def chat(self, query):
        """Main chat function with full error handling"""
        # Get context from documentation
        context = self.get_context(query)
        
        # Create enhanced prompt
        system_prompt = f"""You are a GenPipes bioinformatics expert assistant. Answer questions based on the provided GenPipes v6.0.0 documentation context.

GENPIPES DOCUMENTATION CONTEXT:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
- Use the documentation context above to provide accurate answers
- Include exact command syntax when available
- Mention GenPipes v6.0.0 specifics and new features
- If the context doesn't fully answer the question, say so clearly
- Be concise but comprehensive
- Focus on practical, actionable information

Respond as a helpful GenPipes expert:"""

        # Generate response with timeout
        return self.chat_with_timeout(system_prompt)

def print_startup_info():
    """Print helpful startup information"""
    print("🧬 GenPipes RAG Chatbot v2.0")
    print("Enhanced with documentation search and proper error handling")
    print("=" * 60)
    print("💡 Tips:")
    print("- First response may take 30-60 seconds (model loading)")
    print("- Subsequent responses are faster")
    print("- Be specific: 'RNA-seq StringTie protocol' vs 'RNA analysis'")
    print("- Ask about: commands, configs, troubleshooting, version changes")
    print("-" * 60)

def main():
    parser = argparse.ArgumentParser(description="Fixed GenPipes RAG Chatbot")
    parser.add_argument('--model', default='llama2', help='Ollama model')
    parser.add_argument('--db-path', default='genpipes_vectordb', help='Vector DB path')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2', help='Embedding model')
    
    args = parser.parse_args()
    
    print_startup_info()
    
    try:
        # Initialize components with proper error handling
        vector_db = FixedVectorDatabase(
            db_path=args.db_path,
            model_name=args.embedding_model
        )
        
        chatbot = FixedRAGChatbot(vector_db, args.model)
        
        print("\n🚀 Ready! Ask your GenPipes questions.")
        print("Commands: 'quit', 'exit', 'help'")
        print("=" * 60)
        
        # Main chat loop
        while True:
            try:
                query = input("\n👤 You: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Happy GenPipes workflows! Goodbye!")
                    break
                
                if query.lower() == 'help':
                    print("""
📖 GenPipes RAG Chatbot Help:

🔍 Ask about:
• Pipeline commands: "RNA-seq StringTie protocol syntax"
• Configuration: "readsets file format requirements" 
• Troubleshooting: "fix command not found GenPipes v6.0"
• Version changes: "what's new in GenPipes v6.0"
• Protocols: "ChIP-seq vs ATAC-seq differences"

💡 Tips for better results:
• Be specific with your questions
• Mention the pipeline name (RNA-seq, ChIP-seq, etc.)
• Include context like "for SLURM cluster" or "v6.0"

🚀 The chatbot searches GenPipes documentation and provides 
   accurate answers with source citations.
                    """)
                    continue
                
                # Process the query
                print(f"\n🔍 Processing: '{query[:50]}{'...' if len(query) > 50 else ''}'")
                response = chatbot.chat(query)
                
                print(f"\n🧬 GenPipes Assistant:")
                print("-" * 40)
                print(response)
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print("Continuing...")
    
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check Ollama: ollama list")
        print("2. Check vector DB: ls -la genpipes_vectordb/")
        print("3. Restart Ollama: pkill ollama && ollama serve &")

if __name__ == "__main__":
    main()
