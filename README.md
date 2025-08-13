# Adaptive RAG Chatbot

A fully offline, intelligent document Q&A system with adaptive response capabilities.

## 🚀 Features

- **🧠 Adaptive Response System**: Automatically adjusts answer length based on question complexity
- **🔍 Dual-Source Architecture**: Search both company documents and temporary uploads
- **⚡ Speed Optimized**: Fast responses with intelligent caching and optimized parameters
- **🌐 Fully Offline**: Uses local Ollama models - no internet required
- **📊 ChromaDB Vector Storage**: Persistent vector database for semantic search
- **🎨 Enhanced UI**: Visual indicators showing response type (Short/Medium/Detailed)

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Vector Database**: ChromaDB
- **Language Model**: llama3.2:1b (via Ollama)
- **Embedding Model**: nomic-embed-text (via Ollama)
- **Document Processing**: LangChain + PyMuPDF
- **Architecture**: RAG (Retrieval-Augmented Generation)

## 📋 Prerequisites

1. **Python 3.11+**
2. **Ollama** installed and running
3. Required Ollama models:
   ```bash
   ollama pull llama3.2:1b
   ollama pull nomic-embed-text
   ```

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd BrownieAgent
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Ollama service**
   ```bash
   ollama serve
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open browser** to `http://localhost:8501`

## 📁 Project Structure

```
BrownieAgent/
├── app.py                    # Main adaptive RAG application
├── app_simple.py            # Simple version of the app
├── get_embedding_function.py # Embedding configuration
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
├── data/                   # Put your PDF documents here
├── chroma_vector_db/       # ChromaDB storage (auto-created)
└── test_*.py              # Test files
```

## 💡 How It Works

### Adaptive Response System
The system analyzes your questions and automatically provides:
- **🟢 Short Answers** (200 tokens): For simple questions like "What is X?"
- **🟡 Medium Answers** (350 tokens): For general questions about features, benefits
- **🔵 Detailed Answers** (500 tokens): For complex questions requiring explanation

### Document Sources
- **Company Documents**: Place PDFs in the `data/` folder for permanent access
- **Temporary Documents**: Upload PDFs through the UI for one-time queries

## 🔧 Configuration

### Model Configuration
Edit `get_embedding_function.py` to change:
- Embedding model settings
- Timeout configurations
- Custom model paths

### Response Tuning
Edit `app.py` to adjust:
- Token limits for each response type
- Temperature settings
- Question analysis keywords

## 📊 Performance

- **Response Time**: 2-5 seconds per query
- **Document Processing**: ~1 second per page
- **Memory Usage**: ~2GB with loaded models
- **Offline Operation**: 100% - no internet required

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

- Ollama for local LLM hosting
- ChromaDB for vector storage
- LangChain for document processing
- Streamlit for the web interface
