import requests
from newspaper import Article
import os
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
import re
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="News Article Summarizer & QA", layout="centered")

load_dotenv()

try:
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except KeyError:
    st.error("CRITICAL ERROR: GEMINI_API_KEY environment variable not set. Please set it and restart.")
    st.stop()

@st.cache_resource
def load_gemini_summarizer_model():
    return genai.GenerativeModel('gemini-1.5-flash')

@st.cache_resource
def load_gemini_langchain_chat_model():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.2, convert_system_message_to_human=True)

@st.cache_resource
def load_gemini_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

summarizer_model = load_gemini_summarizer_model()
qa_llm = load_gemini_langchain_chat_model()
embedding_model = load_gemini_embedding_model()

def get_article_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }
    session = requests.Session()
    try:
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        article_obj = Article(url)
        article_obj.download()
        article_obj.parse()
        if not article_obj.text: 
             st.warning(f"Could not extract text from the article at {url}. The content might be dynamically loaded or protected.")
             return None, None, None
        return article_obj.title, article_obj.text, url
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching article: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Error parsing article: {e}")
        return None, None, None

def generate_summary(article_title, article_text, output_language="English"):
    prompt_template = f"""You are an expert assistant that summarizes online articles.
Summarize the following article concisely in {output_language}.

Article Title: {article_title}
Article Text:
{article_text}

Concise Summary in {output_language}:"""
    try:
        response = summarizer_model.generate_content(prompt_template)
        return response.text
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

QA_PROMPT_TEMPLATE = """You are an AI assistant. Your task is to answer questions based ONLY on the provided document excerpts.
If the answer is not found within the excerpts, you MUST state "The answer is not found in the provided article content."
Do not add any information that is not explicitly stated in the excerpts.
Your answer should be concise and directly address the question.
ALWAYS include a "SOURCES:" section in your response, listing the URL(s) from which the information was derived, as provided in the metadata of the excerpts.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:
SOURCES:"""

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE, input_variables=["summaries", "question"]
)

@st.cache_resource(show_spinner="Preparing article for Q&A...")
def create_vector_store_and_qa_chain(_article_text, _source_url, _embedding_model, _llm):
    if not _article_text or not _source_url:
        return None
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs_content = text_splitter.split_text(_article_text)
        
        langchain_docs_with_metadata = [
            {"page_content": doc, "metadata": {"source": _source_url}} for doc in docs_content
        ]
        
        texts_for_faiss = [doc["page_content"] for doc in langchain_docs_with_metadata]
        metadatas_for_faiss = [doc["metadata"] for doc in langchain_docs_with_metadata]

        if not texts_for_faiss:
            st.warning("No text chunks were generated from the article for Q&A.")
            return None

        vector_store = FAISS.from_texts(texts=texts_for_faiss, embedding=_embedding_model, metadatas=metadatas_for_faiss)
        
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=_llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error setting up Q&A module: {e}")
        return None

if 'article_title' not in st.session_state:
    st.session_state.article_title = None
if 'article_text' not in st.session_state:
    st.session_state.article_text = None
if 'article_url' not in st.session_state:
    st.session_state.article_url = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'article_processed' not in st.session_state:
    st.session_state.article_processed = False
if 'current_language' not in st.session_state:
    st.session_state.current_language = "English"


st.title("📰 News Article Summarizer & QA Bot")
st.markdown("Enter a news article URL to get a summary and ask questions about its content.")

url_input = st.text_input("Article URL", key="url_input_field", placeholder="e.g., https://www.example-news.com/article-name")

language_options = [
    "English", "Hindi", "Telugu", "Tamil", "Marathi", "Malayalam", "Kannada",
    "Spanish", "French", "German", "Italian", "Portuguese", "Japanese", "Chinese"
]
selected_language = st.selectbox(
    "Select Output Language for Summary:",
    language_options,
    index=language_options.index(st.session_state.current_language) if st.session_state.current_language in language_options else 0,
    help="Choose the language for the summarized article."
)
st.session_state.current_language = selected_language


if st.button("Process Article", type="primary"):
    if url_input:
        st.session_state.article_title = None
        st.session_state.article_text = None
        st.session_state.article_url = None
        st.session_state.summary = None
        st.session_state.qa_chain = None
        st.session_state.article_processed = False

        with st.spinner(f"Fetching, parsing, and summarizing in {selected_language}..."):
            title, text, url = get_article_content(url_input)

            if title and text and url:
                st.session_state.article_title = title
                st.session_state.article_text = text
                st.session_state.article_url = url
                st.session_state.summary = generate_summary(title, text, output_language=selected_language)
                st.session_state.article_processed = True
                st.session_state.qa_chain = create_vector_store_and_qa_chain(
                    st.session_state.article_text,
                    st.session_state.article_url,
                    embedding_model,
                    qa_llm
                )
                if not st.session_state.qa_chain and st.session_state.article_text:
                    st.warning("Q&A module could not be initialized for this article, though summary might be available.")
            elif url_input and not (title and text and url): 
                 st.error("Failed to retrieve or parse article content. Please check the URL or try another article.")
            else:
                 st.warning("Could not process the article. Please check the URL.")
    else:
        st.warning("Please enter an article URL.")

if st.session_state.article_processed and st.session_state.summary:
    st.markdown("---")
    st.subheader("Original Article Details:")
    st.write(f"**Title:** {st.session_state.article_title}")
    st.write(f"**Source URL:** [{st.session_state.article_url}]({st.session_state.article_url})")

    st.subheader(f"Generated Summary ({st.session_state.current_language}):")
    st.write(st.session_state.summary)

if st.session_state.article_processed and st.session_state.qa_chain:
    st.markdown("---")
    st.subheader(f"Ask a Question about: '{st.session_state.article_title}'")
    qa_query = st.text_input("Your question:", key="qa_query_input", placeholder="e.g., What are the key findings?")

    if st.button("Get Answer", key="get_answer_button"):
        if qa_query:
            with st.spinner("Searching for the answer..."):
                try:
                    result = st.session_state.qa_chain({"question": qa_query})
                    st.markdown("#### Answer:")
                    answer_text = result.get("answer", "No answer could be generated.").strip()
                    
                    if "FINAL ANSWER:" in answer_text: 
                        answer_text = answer_text.split("FINAL ANSWER:", 1)[-1].strip()
                    if "SOURCES:" in answer_text: 
                        answer_text = answer_text.split("SOURCES:", 1)[0].strip()

                    st.write(answer_text if answer_text else "No answer found in the text.")
                    
                    st.markdown("#### Sources:")
                    sources_text = result.get("sources", "").strip()
                    if sources_text:
                        source_urls = list(set(s.strip() for s in sources_text.split(',') if s.strip()))
                        if source_urls:
                            for src_url in source_urls:
                                st.markdown(f"- [{src_url}]({src_url})")
                        else:
                             st.write(f"The article URL: [{st.session_state.article_url}]({st.session_state.article_url})")
                    else:
                        st.write(f"The article URL: [{st.session_state.article_url}]({st.session_state.article_url}) (General source for context)")

                except Exception as e:
                    st.error(f"Error getting answer: {e}")
        else:
            st.warning("Please type a question.")
elif st.session_state.article_processed and not st.session_state.qa_chain and st.session_state.article_text:
    st.markdown("---")
    st.info("Q&A is not available for this article because the Q&A module could not be initialized.")


st.markdown("---")
