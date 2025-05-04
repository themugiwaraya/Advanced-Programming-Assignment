from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
import chromadb
from duckduckgo_search import DDGS
import re 
import streamlit as st

st.set_page_config(page_title="Web Search Chatbot", layout="wide")
st.title("Web Search Chatbot")

def init_chroma():
    return chromadb.Client()

def extract_keywords(query):
    try:
        llm = OllamaLLM(model="llama3.2")
        prompt = f"Extract only the most relevant search query without any extra words:\n\n{query}\n\nOnly return the cleaned search query:"
        response = llm.invoke(prompt).strip()
        keywords = re.sub(r"[^a-zA-Z0-9 ]", "", response)

        return keywords
    except Exception as e:
        return str(e)

def search_web(keywords, max_results=10):
    try:
        with DDGS() as ddgs:
            results = ddgs.text(keywords, max_results=max_results)
            urls = [result['href'] for result in results]
        return urls
    except Exception as e:
        st.error(f"Ошибка при поиске: {e}")
        return []

def scrape_content(urls):
    try:
        loader = AsyncHtmlLoader(urls)
        try:
            docs = loader.load()
        except Exception as e:
            st.warning(f"Ошибка при загрузке документов: {e}")
            return []

        if not docs:
            st.error("Не удалось загрузить контент с указанных URL.")
            return []

        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = []
        for doc in docs:
            try:
                transformed = bs_transformer.transform_documents(
                    [doc],
                    tags_to_extract=["p", "h1", "h2", "h3", "article"],
                    remove_unwanted_tags=True
                )
                docs_transformed.extend(transformed)
            except Exception as e:
                st.warning(f"Ошибка при обработке документа: {e}")

        cleaned_docs = []
        for doc in docs_transformed:
            text = re.sub(r'\s+', ' ', doc.page_content)
            text = text[:4000]  
            cleaned_docs.append(text)

        return cleaned_docs
    except Exception as e:
        st.error(f"Ошибка при обработке контента: {e}")
        return []

def create_vectorstore(texts):
    embeddings = OllamaEmbeddings(model="llama3.2")
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory="./chroma_db" 
    )
    return vectorstore

def create_llm_prompt(llm_query, search_results):
    context = "\n\n---\n\n".join(search_results)
    prompt = f"""
    Answer the question using only the context below.

    Context:
    {context}

    Question: {llm_query}

    Answer:
    """
    return prompt

def generate_response(prompt):
    try:
        llm = OllamaLLM(model="llama3.2")
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        st.error(f"Ошибка при генерации ответа: {e}")
        return "Извините, произошла ошибка при генерации ответа."

with st.form("query_form"):
    user_query = st.text_area("Введите запрос для поиска в интернете:", "")
    submitted = st.form_submit_button("Отправить запрос")

if submitted:
    if user_query:
        st.info("Извлечение ключевых слов...")
        keywords = extract_keywords(user_query)
        st.info(keywords)

        if keywords:
            st.info("Идет поиск в интернете...")
            urls = search_web(keywords)
            contents = scrape_content(urls)

            if contents:
                st.success("Поиск завершен. Найдена информация.")

                vectorstore = create_vectorstore(contents)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                relevant_docs = retriever.get_relevant_documents(user_query)

                context = "\n".join([doc.page_content for doc in relevant_docs])
                st.info("Генерация ответа...")
                prompt = create_llm_prompt(user_query, contents)
                response = generate_response(prompt)

                st.subheader("Ответ LLM:")
                st.write(response)

                with st.expander("Источники"):
                    for url in urls:
                        st.markdown(f"- {url}")
            else:
                st.error("Не удалось найти полезную информацию.")
        else:
            st.warning("Не удалось извлечь ключевые слова из запроса.")
    else:
        st.warning("Пожалуйста, введите запрос для поиска.")