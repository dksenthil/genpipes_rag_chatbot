#!/usr/bin/env python3
"""
GenPipes Documentation Scraper and Vector Database Creator
Scrapes GenPipes documentation and creates embeddings for RAG chatbot
"""

import os
import sys
import json
import time
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import hashlib
from pathlib import Path
import pickle
from typing import List, Dict, Any
import argparse
from datetime import datetime

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    print("Missing dependencies. Install with:")
    print("pip install chromadb sentence-transformers beautifulsoup4 requests numpy")
    sys.exit(1)

class GenPipesDocScraper:
    def __init__(self, base_url="https://genpipes.readthedocs.io/en/genpipes-v6.0.0/", 
                 output_dir="genpipes_docs", cache_dir="cache"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.visited_urls = set()
        self.scraped_content = []
        self.session = requests.Session()
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Configure session
        self.session.headers.update({
            'User-Agent': 'GenPipes Documentation Scraper 1.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        })
    
    def is_valid_url(self, url):
        """Check if URL should be scraped"""
        if not url.startswith(self.base_url):
            return False
        
        # Skip certain file types and fragments
        skip_patterns = [
            '.pdf', '.zip', '.tar.gz', '.jpg', '.png', '.gif',
            '#', 'mailto:', 'javascript:', '_static/', '_images/',
            'genindex.html', 'search.html'
        ]
        
        return not any(pattern in url for pattern in skip_patterns)
    
    def get_page_cache_path(self, url):
        """Get cache file path for a URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.html"
    
    def fetch_page(self, url, use_cache=True):
        """Fetch a web page with caching"""
        cache_path = self.get_page_cache_path(url)
        
        # Use cache if available and recent
        if use_cache and cache_path.exists():
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age < 24 * 3600:  # 24 hours
                print(f"📋 Using cached: {url}")
                return cache_path.read_text(encoding='utf-8')
        
        try:
            print(f"🌐 Fetching: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Cache the response
            cache_path.write_text(response.text, encoding='utf-8')
            time.sleep(1)  # Be respectful
            
            return response.text
            
        except Exception as e:
            print(f"❌ Error fetching {url}: {e}")
            return None
    
    def extract_content(self, html, url):
        """Extract meaningful content from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 
                           'aside', '.sidebar', '.toctree-wrapper']):
            element.decompose()
        
        # Extract main content
        main_content = (
            soup.find('div', class_='document') or
            soup.find('main') or
            soup.find('div', class_='body') or
            soup.find('div', class_='content') or
            soup.body
        )
        
        if not main_content:
            return None
        
        # Extract title
        title = (
            soup.find('h1') or 
            soup.find('title')
        )
        title_text = title.get_text().strip() if title else "No Title"
        
        # Extract text content
        text_content = main_content.get_text()
        
        # Clean up text
        lines = [line.strip() for line in text_content.split('\n')]
        clean_lines = [line for line in lines if line and len(line) > 10]
        clean_text = '\n'.join(clean_lines)
        
        # Extract code blocks
        code_blocks = []
        for code in main_content.find_all(['code', 'pre']):
            code_text = code.get_text().strip()
            if len(code_text) > 20:  # Only meaningful code blocks
                code_blocks.append(code_text)
        
        # Extract links
        internal_links = []
        for link in main_content.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(url, href)
            if self.is_valid_url(full_url):
                internal_links.append(full_url)
        
        return {
            'url': url,
            'title': title_text,
            'content': clean_text,
            'code_blocks': code_blocks,
            'internal_links': internal_links,
            'scraped_at': datetime.now().isoformat(),
            'content_length': len(clean_text),
            'word_count': len(clean_text.split())
        }
    
    def scrape_documentation(self, max_pages=100):
        """Scrape GenPipes documentation"""
        print(f"🧬 Starting GenPipes documentation scraping...")
        print(f"Base URL: {self.base_url}")
        print(f"Max pages: {max_pages}")
        
        # Start with main page
        to_visit = [self.base_url]
        scraped_count = 0
        
        while to_visit and scraped_count < max_pages:
            url = to_visit.pop(0)
            
            if url in self.visited_urls:
                continue
            
            self.visited_urls.add(url)
            
            # Fetch and parse page
            html = self.fetch_page(url)
            if not html:
                continue
            
            content_data = self.extract_content(html, url)
            if not content_data:
                continue
            
            # Filter out pages with minimal content
            if content_data['word_count'] < 50:
                print(f"⏭️  Skipping low-content page: {url}")
                continue
            
            self.scraped_content.append(content_data)
            scraped_count += 1
            
            print(f"✅ [{scraped_count}/{max_pages}] {content_data['title'][:50]}...")
            
            # Add new internal links to visit
            for link in content_data['internal_links']:
                if link not in self.visited_urls and link not in to_visit:
                    to_visit.append(link)
        
        print(f"📚 Scraped {len(self.scraped_content)} pages")
        
        # Save scraped data
        output_file = self.output_dir / 'scraped_content.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_content, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved to: {output_file}")
        return self.scraped_content

class VectorDatabase:
    def __init__(self, db_path="genpipes_vectordb", model_name="all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model_name = model_name
        
        # Initialize embedding model
        print(f"🤖 Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection_name = "genpipes_docs"
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
            print(f"📋 Using existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "GenPipes documentation embeddings"}
            )
            print(f"🆕 Created new collection: {self.collection_name}")
    
    def chunk_text(self, content, chunk_size=500, overlap=50):
        """Split text into overlapping chunks"""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 100:  # Skip very short chunks
                chunks.append(chunk)
        
        return chunks
    
    def create_embeddings(self, scraped_content):
        """Create and store embeddings for scraped content"""
        print("🔄 Creating embeddings...")
        
        documents = []
        metadatas = []
        ids = []
        
        for idx, page_data in enumerate(scraped_content):
            # Chunk the content
            chunks = self.chunk_text(page_data['content'])
            
            for chunk_idx, chunk in enumerate(chunks):
                doc_id = f"page_{idx}_chunk_{chunk_idx}"
                
                documents.append(chunk)
                metadatas.append({
                    'url': page_data['url'],
                    'title': page_data['title'],
                    'page_idx': idx,
                    'chunk_idx': chunk_idx,
                    'scraped_at': page_data['scraped_at'],
                    'content_type': 'text'
                })
                ids.append(doc_id)
            
            # Add code blocks separately
            for code_idx, code in enumerate(page_data['code_blocks']):
                if len(code) > 50:  # Only meaningful code blocks
                    doc_id = f"page_{idx}_code_{code_idx}"
                    
                    documents.append(code)
                    metadatas.append({
                        'url': page_data['url'],
                        'title': page_data['title'],
                        'page_idx': idx,
                        'code_idx': code_idx,
                        'scraped_at': page_data['scraped_at'],
                        'content_type': 'code'
                    })
                    ids.append(doc_id)
        
        print(f"📊 Processing {len(documents)} chunks...")
        
        # Create embeddings in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(batch_docs).tolist()
            
            # Add to collection
            self.collection.add(
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=batch_metas,
                ids=batch_ids
            )
            
            print(f"✅ Processed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
        
        print(f"💾 Stored {len(documents)} embeddings in vector database")
    
    def search(self, query, n_results=5):
        """Search for relevant documents"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        return results

class RAGChatbot:
    def __init__(self, vector_db, ollama_model="llama2"):
        self.vector_db = vector_db
        self.ollama_model = ollama_model
    
    def get_context(self, query, n_results=3):
        """Get relevant context for a query"""
        results = self.vector_db.search(query, n_results=n_results)
        
        context_parts = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            context_parts.append(f"""
--- Source {i+1}: {metadata['title']} ---
URL: {metadata['url']}
Content Type: {metadata['content_type']}
Relevance Score: {1-distance:.3f}

{doc[:500]}{'...' if len(doc) > 500 else ''}
""")
        
        return '\n'.join(context_parts)
    
    def chat(self, query):
        """Chat with RAG-enhanced responses"""
        try:
            import ollama
            
            # Get relevant context
            context = self.get_context(query)
            
            # Create enhanced prompt
            system_prompt = f"""You are a GenPipes expert assistant. Use the following documentation context to answer questions accurately.

CONTEXT FROM GENPIPES DOCUMENTATION:
{context}

Based on this context, provide accurate and helpful answers about GenPipes workflows, commands, and troubleshooting. If the context doesn't contain relevant information, say so clearly.

Rules:
- Always cite specific sources when possible
- Provide exact command syntax from the documentation
- Mention version information when relevant
- If unsure, recommend checking the official documentation
"""
            
            # Get response from Ollama
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]
            )
            
            return response['message']['content']
            
        except Exception as e:
            return f"Error generating response: {e}"

def main():
    parser = argparse.ArgumentParser(description="GenPipes Documentation Scraper and Vector DB")
    parser.add_argument('--scrape', action='store_true', help='Scrape documentation')
    parser.add_argument('--vectorize', action='store_true', help='Create vector database')
    parser.add_argument('--chat', action='store_true', help='Start RAG chatbot')
    parser.add_argument('--max-pages', type=int, default=100, help='Max pages to scrape')
    parser.add_argument('--model', default='llama2', help='Ollama model for chat')
    parser.add_argument('--base-url', default='https://genpipes.readthedocs.io/en/genpipes-v6.0.0/', 
                       help='Base URL to scrape')
    
    args = parser.parse_args()
    
    if args.scrape:
        scraper = GenPipesDocScraper(base_url=args.base_url)
        scraped_content = scraper.scrape_documentation(max_pages=args.max_pages)
        print(f"✅ Scraping complete: {len(scraped_content)} pages")
    
    if args.vectorize:
        # Load scraped content
        content_file = Path('genpipes_docs/scraped_content.json')
        if content_file.exists():
            with open(content_file, 'r', encoding='utf-8') as f:
                scraped_content = json.load(f)
            
            # Create vector database
            vector_db = VectorDatabase()
            vector_db.create_embeddings(scraped_content)
            print("✅ Vector database creation complete")
        else:
            print("❌ No scraped content found. Run with --scrape first.")
    
    if args.chat:
        try:
            # Load vector database
            vector_db = VectorDatabase()
            chatbot = RAGChatbot(vector_db, args.model)
            
            print(f"🧬 GenPipes RAG Chatbot (Model: {args.model})")
            print("Type 'quit' to exit")
            print("-" * 50)
            
            while True:
                query = input("\n👤 You: ").strip()
                if query.lower() in ['quit', 'exit']:
                    break
                
                if query:
                    print("🧬 GenPipes Assistant: ", end="", flush=True)
                    response = chatbot.chat(query)
                    print(response)
        
        except Exception as e:
            print(f"❌ Error starting chatbot: {e}")
    
    if not any([args.scrape, args.vectorize, args.chat]):
        print("Usage examples:")
        print("python genpipes_scraper.py --scrape --max-pages 50")
        print("python genpipes_scraper.py --vectorize")
        print("python genpipes_scraper.py --chat --model llama2")
        print("python genpipes_scraper.py --scrape --vectorize --chat")

if __name__ == "__main__":
    main()
