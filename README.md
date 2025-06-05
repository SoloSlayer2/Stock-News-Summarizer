# 📈 Stock News Summarizer using LangChain + Ollama + Streamlit

This app summarizes the latest news from URLs provided by the user, specifically targeting financial or stock-related insights. It uses Retrieval-Augmented Generation (RAG) to generate concise bullet-point summaries of news articles, helping you understand market-impacting developments quickly.

---

## 🔧 Features

- 🔗 Load multiple financial news articles via URLs
- 📄 Extract and chunk text using LangChain's `UnstructuredURLLoader`
- 🧠 Embed documents using `OllamaEmbeddings` and store in FAISS vector store
- 🤖 Retrieve relevant context with semantic search
- 📝 Generate bullet-point summaries using `Mistral` model via `ChatOllama`
- 📌 Display source links (citations) for traceability
- 🧹 Reset the app with one click

---
## 🛠️ Requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed and running
  ```bash
  ollama run mistral
  ollama run nomic-embed-text
