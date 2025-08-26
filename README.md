# GenPipes RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built for exploring and querying the **GenPipes v6.0.0** documentation.  
It combines:  

- **Scraper**: downloads and indexes GenPipes docs.  
- **Vector Database**: powered by [ChromaDB](https://www.trychroma.com/) + [sentence-transformers](https://www.sbert.net/).  
- **Chat Interface**: integrates with [Ollama](https://ollama.ai) for local LLM inference (e.g., Llama2).  

---

## 🚀 Features
- Scrape GenPipes docs (configurable pages).  
- Vectorize scraped text for efficient retrieval.  
- Chat with a local LLM (via Ollama).  
- Debug utilities for model + database testing.  

---

## 📦 Installation

Clone the repo:
```bash
git clone https://github.com/dksenthil/genpipes_rag_chatbot.git
cd genpipes_rag_chatbot
```

Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate     # macOS/Linux
# venv\Scripts\activate    # Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

(If you run into `numpy` issues, install a compatible version:  
`pip install "numpy<2.0" --force-reinstall`)

---

## ⚡ Usage

### 1. Scrape documentation
```bash
python genpipes_scraper.py --scrape --max-pages 50
```

### 2. Vectorize the documents
```bash
python genpipes_scraper.py --vectorize
```

### 3. Start the chatbot (Llama2 via Ollama)
Make sure Ollama is installed and running:
```bash
ollama serve &
```

Then:
```bash
python genpipes_scraper.py --chat --model llama2
```

Example:
```
User: What pipelines are available in GenPipes?
Bot: GenPipes v6.0.0 provides workflows for RNA-seq, WGS, WGBS, ChIP-seq...
```

---

## 🛠️ Utilities
- `run_setup_chatbot.sh` → setup helper script.  
- `run_scraper.sh` → quick scraping workflow.  
- `working_genpipes_chat.py` → stable chat entrypoint.  
- `debug.py` → for testing Ollama connectivity.  

---

## 🧪 Example Ollama Test
```bash
ollama run llama2 "Say hello"
```

---

## 📜 License
MIT License. See [LICENSE](LICENSE) for details.  
