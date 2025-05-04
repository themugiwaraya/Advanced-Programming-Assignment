import streamlit as st
import tempfile
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.schema import ChatMessage
from langchain_core.messages import HumanMessage, AIMessage
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import logging
import json
import os
import io
import base64

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_contexts' not in st.session_state:
    st.session_state.chat_contexts = {}

if not os.path.exists("./chroma_db"):
    os.makedirs("./chroma_db")

if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(allow_reset=True, is_persistent=True)
    )
    embeddings = OllamaEmbeddings(model="llama3.2")
    st.session_state.collection = Chroma(
        client=st.session_state.chroma_client,
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

def save_documents():
    try:
        with open("saved_documents.json", "w") as f:
            json.dump(st.session_state.documents, f)
    except Exception as e:
        st.error(f"Error saving documents: {e}")

def load_documents():
    try:
        with open("saved_documents.json", "r") as f:
            loaded_docs = json.load(f)
            # Re-add documents to ChromaDB
            for doc_name, text_content in loaded_docs.items():
                add_to_chroma(text_content, doc_name)
            return loaded_docs
    except FileNotFoundError:
        return {}

def load_chat_history():
    try:
        with open("chat_history.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_chat_history(chat_contexts):
    with open("chat_history.json", "w") as f:
        json.dump(chat_contexts, f)

def stream_chat(model, messages, context_docs):
    try:
        llm = OllamaLLM(model=model, request_timeout=120.0)
        prompt = create_llm_prompt(messages[-1].content, context_docs)
        response_placeholder = st.empty()
        response = ""
        
        for chunk in llm.stream(prompt):
            response += chunk
            response_placeholder.markdown(response)
        
        return response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        raise e

def process_document(file, file_type):
    """Process uploaded document and return text content"""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.getvalue())
            file_path = tmp_file.name

        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            text = " ".join([page.page_content for page in pages])
        else:
            loader = TextLoader(file_path)
            documents = loader.load()
            text = documents[0].page_content

        Path(file_path).unlink()
        return text
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return None

def add_to_chroma(text, document_name):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        st.session_state.collection.add_texts(
            texts=chunks,
            metadatas=[{"source": document_name} for _ in chunks]
        )
        return True
    except Exception as e:
        st.error(f"Error adding to ChromaDB: {e}")
        return False

def delete_document(doc_name):
    try:
        # Delete from ChromaDB
        st.session_state.collection.delete(
            where={"source": doc_name}
        )
        # Delete from session state
        del st.session_state.documents[doc_name]
        # Delete associated chat context
        if doc_name in st.session_state.chat_contexts:
            del st.session_state.chat_contexts[doc_name]
            save_chat_history(st.session_state.chat_contexts)
        # Clear current doc if deleted document was being chatted with
        if hasattr(st.session_state, 'current_doc') and st.session_state.current_doc == doc_name:
            del st.session_state.current_doc
        
        save_documents()
        return True
    except Exception as e:
        st.error(f"Error deleting document: {e}")
        return False

def query_documents(query, n_results=5):
    try:
        docs = st.session_state.collection.similarity_search(
            query,
            k=n_results
        )
        return [doc.page_content for doc in docs]
    except Exception as e:
        st.error(f"Error querying documents: {e}")
        return None

def scrape_content(url):
    try:
        loader = AsyncHtmlLoader([url])
        docs = loader.load()
        
        if not docs:
            st.error("Failed to load content from URL.")
            return None

        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(
            docs,
            tags_to_extract=["p", "h1", "h2", "h3", "article"],
            remove_unwanted_tags=True
        )

        text = " ".join([re.sub(r'\s+', ' ', doc.page_content) for doc in docs_transformed])
        return text[:4000]
    except Exception as e:
        st.error(f"Error processing URL: {e}")
        return None

def create_llm_prompt(query, context_docs):
    context = "\n\n---\n\n".join(context_docs)
    return f"""
    Using only the context provided below, answer the following question.
    If the answer cannot be found in the context, say "I don't have enough information to answer that question."

    Context:
    {context}

    Question: {query}

    Answer:
    """

def generate_visualization_html(text):
    try:
        # Word Cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        # Matplotlib figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        ax1.imshow(wordcloud, interpolation='bilinear')
        ax1.axis('off')
        ax1.set_title('Word Cloud')

        words = text.lower().split()
        word_freq = defaultdict(int)
        for word in words:
            if len(word) > 3:
                word_freq[word] += 1
        
        top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10])
        
        ax2.bar(top_words.keys(), top_words.values())
        ax2.set_xticklabels(top_words.keys(), rotation=45)
        ax2.set_title('Top 10 Word Frequencies')
        
        plt.tight_layout()

        # Save plot to a buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)

        return f'<img src="data:image/png;base64,{image_base64}" alt="Document Visualization">'
    except Exception as e:
        st.error(f"Error generating visualizations: {e}")
        return None

def main():
    st.title("Document Analysis and Chat System")
    
    # Load saved chat history
    if 'chat_contexts' not in st.session_state:
        st.session_state.chat_contexts = load_chat_history()
    
    # Load saved documents
    if 'documents' not in st.session_state:
        st.session_state.documents = load_documents() or {}
    
    # Sidebar for document management
    with st.sidebar:
        st.header("Document Management")
        
        # File upload
        uploaded_file = st.file_uploader("Upload PDF or Text file", type=['pdf', 'txt'])
        if uploaded_file:
            file_type = uploaded_file.type.split('/')[-1]
            text_content = process_document(uploaded_file, file_type)
            if text_content:
                st.session_state.documents[uploaded_file.name] = text_content
                if add_to_chroma(text_content, uploaded_file.name):
                    st.success(f"Successfully added {uploaded_file.name}")
                    save_documents()
        
        # URL input with button
        col1, col2 = st.columns([3, 1])
        with col1:
            url_input = st.text_input("Enter URL:")
        with col2:
            if st.button("Add URL"):
                if url_input:
                    scraped_content = scrape_content(url_input)
                    if scraped_content:
                        doc_name = f"url_{len(st.session_state.documents)}"
                        st.session_state.documents[doc_name] = scraped_content
                        if add_to_chroma(scraped_content, doc_name):
                            st.success("Successfully added URL content")
                            save_documents()
        
        # Document list with chat buttons
        st.subheader("Available Documents")
        for doc_name in list(st.session_state.documents.keys()):
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"Chat: {doc_name}", key=f"chat_{doc_name}"):
                    st.session_state.current_doc = doc_name
            with col2:
                if st.button("Delete", key=f"del_{doc_name}"):
                    if delete_document(doc_name):
                        st.rerun()

    # Main chat area
    if hasattr(st.session_state, 'current_doc'):
        st.subheader(f"Chat about: {st.session_state.current_doc}")
        
        # Display chat history and visualization
        if st.session_state.current_doc not in st.session_state.chat_contexts:
            st.session_state.chat_contexts[st.session_state.current_doc] = []
        
        # Add visualization to the first message
        if len(st.session_state.chat_contexts[st.session_state.current_doc]) == 0:
            doc_text = st.session_state.documents[st.session_state.current_doc]
            viz_html = generate_visualization_html(doc_text)
            if viz_html:
                viz_message = {
                    "role": "assistant", 
                    "content": f"Document Visualization:\n\n{viz_html}",
                    "type": "visualization"
                }
                st.session_state.chat_contexts[st.session_state.current_doc].append(viz_message)
        
        for message in st.session_state.chat_contexts[st.session_state.current_doc]:
            with st.chat_message(message["role"]):
                if message.get("type") == "visualization":
                    st.markdown(message["content"], unsafe_allow_html=True)
                else:
                    st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Your question"):
            # Add user message
            user_message = {"role": "user", "content": prompt}
            st.session_state.chat_contexts[st.session_state.current_doc].append(user_message)
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                context_docs = query_documents(prompt)
                if context_docs:
                    start_time = time.time()
                    try:
                        messages = [ChatMessage(role=msg["role"], content=msg["content"]) 
                                  for msg in st.session_state.chat_contexts[st.session_state.current_doc]]
                        response = stream_chat("llama3.2", messages, context_docs)
                        duration = time.time() - start_time
                        
                        # Add assistant message
                        assistant_message = {
                            "role": "assistant",
                            "content": f"{response}\n\nDuration: {duration:.2f} seconds"
                        }
                        st.session_state.chat_contexts[st.session_state.current_doc].append(assistant_message)
                        
                        # Save chat history
                        save_chat_history(st.session_state.chat_contexts)
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()