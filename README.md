# Simple RAG System

A minimal, local **Retrieval-Augmented Generation (RAG)** system built with [LangChain](https://python.langchain.com/), [Ollama](https://ollama.com/) and [Chroma](https://www.trychroma.com/).

The system loads text documents from `data/documents/`, splits them into chunks, embeds them with a local Ollama embedding model, stores them in a persistent Chroma vector database, and answers questions using a local LLM. **No API keys and no cloud services are required** — everything runs on your machine.

## How it works

```
Text documents
      │
      ▼
TextLoader  →  RecursiveCharacterTextSplitter  →  OllamaEmbeddings
      │                                                  │
      ▼                                                  ▼
                       Chroma (persistent vector store)
                                  │
                                  ▼  (retrieve top 3 chunks)
                        ChatPromptTemplate + ChatOllama
                                  │
                                  ▼
                               Answer
```

The prompt instructs the model to answer **only** from the retrieved context, and to say *"I don't know based on the provided documents."* when the answer is not in the documents (grounded answers, fewer hallucinations).

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com/download) installed and running locally

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/kfn-m/simple-rag-system.git
   cd simple-rag-system
   ```

2. Create and activate a virtual environment:

   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS / Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Pull the Ollama models used by the project:

   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen3:4b
   ```

   > Make sure the Ollama server is running (`ollama serve`) before running the app.

## Usage

Run the script from the project root (the document path is relative to it):

```bash
python app.py
```

Example output:

```
Documents loaded: 1
Chunks created: 4
Embedding model ready!
Documents added to vector database.
Retriever ready!
Qwen3 ready!

Question: What is RAG?
Answer: RAG (Retrieval-Augmented Generation) is a technique that lets an AI
system retrieve relevant information from external documents before generating
an answer...
```

The first run creates a `chroma_db/` folder and indexes the documents. Later runs reuse the existing vector database, so indexing is only done once.

To ask a different question, change the `question` variable at the bottom of `app.py`, or call the function yourself:

```python
from app import ask_rag

print(ask_rag("What are the steps of a RAG pipeline?"))
```

## Using your own documents

1. Add a `.txt` file inside `data/documents/`.
2. Update the path passed to `TextLoader` in `app.py`.
3. Delete the `chroma_db/` folder (or add documents via `vector_store.add_documents(...)`) so the new content is indexed.

You can also tune the chunking and retrieval settings in `app.py`:

| Setting | Current value | Location |
| --- | --- | --- |
| `chunk_size` / `chunk_overlap` | `200` / `50` | `RecursiveCharacterTextSplitter` |
| Retrieved chunks (`k`) | `3` | `vector_store.as_retriever(...)` |
| Embedding model | `nomic-embed-text` | `OllamaEmbeddings` |
| LLM | `qwen3:4b`, `temperature=0.3` | `ChatOllama` |

## Project structure

```
.
├── app.py                          # Full RAG pipeline + example question
├── requirements.txt                # Python dependencies
├── data/
│   └── documents/
│       └── rag_notes.txt           # Sample knowledge base
├── research/
│   └── rag-demo.ipynb              # Step-by-step notebook version
└── chroma_db/                      # Generated vector store (git-ignored)
```

## License

MIT — feel free to use and modify it.
