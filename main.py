import streamlit as st
from llama_index.core.llms import ChatMessage
import logging
import time
from sentence_transformers import SentenceTransformer
from llama_index.llms.ollama import Ollama
import chromadb
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader

logging.basicConfig(level=logging.INFO)

# Создаем клиента для взаимодействия с ChromaDB
chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))

# Загружаем предварительно обученную модель SentenceTransformer для создания эмбеддингов
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Определяем коллекцию в ChromaDB
collection_name = "rag_collection_demo"
try:
    collection = chroma_client.get_collection(name=collection_name)
    if collection.metadata.get("dimension") != 384:
        chroma_client.delete_collection(name=collection_name)
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"description": "A collection for RAG with Ollama", "dimension": 384}
        )
except chromadb.errors.CollectionNotFoundError:
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"description": "A collection for RAG with Ollama", "dimension": 384}
    )

# Определяем модель LLM для использования
llm_model = "llama3.2"

# Генерация эмбеддингов для текстов
def generate_embeddings(texts):
    if isinstance(texts, str):
        texts = [texts]
    return embedding_model.encode(texts, convert_to_numpy=True).tolist()

# Добавление документов в коллекцию ChromaDB
def add_documents_to_collection(documents, ids):
    embeddings = generate_embeddings(documents)
    collection.add(
        documents=documents,
        ids=ids,
        embeddings=embeddings
    )

# Поиск в ChromaDB по смысловому сходству
def query_chromadb(query_text, n_results=1):
    query_embedding = generate_embeddings(query_text)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results["documents"], results["metadatas"]

# Удаление всех документов из коллекции ChromaDB
def delete_all_documents():
    all_docs = collection.get()
    if all_docs["ids"]:
        collection.delete(ids=all_docs["ids"])
        st.success("All documents deleted successfully!")
    else:
        st.warning("No documents to delete.")

# Взаимодействие с Ollama через библиотеку

def query_ollama(prompt):
    llm = Ollama(model=llm_model, request_timeout=120.0)
    messages = [ChatMessage(role="user", content=prompt)]
    response = llm.stream_chat(messages)
    return "".join([r.delta for r in response])

# Процесс RAG с использованием ChromaDB и Ollama
def rag_pipeline(query_text):
    retrieved_docs, metadata = query_chromadb(query_text)
    context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."
    augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    response = query_ollama(augmented_prompt)
    return response

# Проверяем, существует ли список сообщений
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Извлечение текста из файла
def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type == 'pdf':
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            loader = PyPDFLoader(temp_file.name)
            documents = loader.load()
        os.unlink(temp_file.name)  # Удаление временного файла
        return [doc.page_content for doc in documents]
    elif file_type == 'txt':
        return [uploaded_file.read().decode('utf-8')]
    else:
        st.error("Unsupported file type. Please upload a PDF or TXT file.")
        return []

# Основная функция Streamlit-приложения
def main():
    st.title("Interactive RAG with Ollama")

    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file:
        documents = extract_text_from_file(uploaded_file)
        if documents:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]
            add_documents_to_collection(documents, doc_ids)
            st.success("File content added to ChromaDB!")

    if st.button("Show All Documents"):
        all_docs = collection.get()
        st.write("Documents in ChromaDB:")
        st.write(all_docs["documents"] if all_docs["documents"] else "No documents found.")

    if st.button("Delete All Documents"):
        delete_all_documents()

    with st.form("add_document_form"):
        new_doc = st.text_area("Add a new document:")
        new_doc_id = st.text_input("Document ID:")
        submitted = st.form_submit_button("Add Document")
        if submitted and new_doc and new_doc_id:
            add_documents_to_collection([new_doc], [new_doc_id])
            st.success("Document added successfully!")

    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                start_time = time.time()
                with st.spinner("Generating response..."):
                    try:
                        response_message = rag_pipeline(prompt)
                        duration = time.time() - start_time
                        response_message_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"
                        st.session_state.messages.append({"role": "assistant", "content": response_message_with_duration})
                        st.write(response_message_with_duration)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()