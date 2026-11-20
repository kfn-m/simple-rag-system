from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# ==========================================
# 1. Load Documents
# ==========================================

loader = TextLoader("data/documents/rag_notes.txt")

documents = loader.load()

print("Documents loaded:", len(documents))


# ==========================================
# 2. Split Documents
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Chunks created:", len(chunks))


# ==========================================
# 3. Embeddings
# ==========================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("Embedding model ready!")


# ==========================================
# 4. Vector Database
# ==========================================

vector_store = Chroma(
    collection_name="rag_collection",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)

# Add documents only if the database is empty
if vector_store._collection.count() == 0:
    vector_store.add_documents(chunks)
    print("Documents added to vector database.")
else:
    print("Using existing vector database.")


# ==========================================
# 5. Retriever
# ==========================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

print("Retriever ready!")


# ==========================================
# 6. Local LLM
# ==========================================

llm = ChatOllama(
    model="qwen3:4b",
    temperature=0.3
)

print("Qwen3 ready!")


# ==========================================
# 7. Prompt
# ==========================================

prompt = ChatPromptTemplate.from_template("""

You are a helpful AI assistant.

Answer the question using only the context provided below.

If the answer cannot be found in the context, say:

"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
""")


# ==========================================
# 8. RAG Function
# ==========================================

def ask_rag(question):

    relevant_chunks = retriever.invoke(question)

    context = "\n\n".join(
        chunk.page_content  
        for chunk in relevant_chunks
    )

    messages = prompt.invoke({
        "context": context,
        "question": question
    })

    response = llm.invoke(messages)

    return response.content


# ==========================================
# 9. Test
# ==========================================

question = "What is RAG?"

answer = ask_rag(question)

print("\nQuestion:", question)
print("\nAnswer:", answer) 