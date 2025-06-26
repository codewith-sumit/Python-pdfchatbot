import os
import stat
import hashlib
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from pinecone import Pinecone

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Validate API keys
if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    st.error("API keys are missing. Please check your .env file.")
    st.stop()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "pdf-chatbot-index"
index = pc.Index(INDEX_NAME)

# Utility functions
def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "".join(page.extract_text() or "" for page in reader.pages)

def split_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=1000)
    return splitter.split_text(text)

def load_vector_data(chunks, file_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GOOGLE_API_KEY)
    file_hash = hashlib.sha256(file_name.encode()).hexdigest()
    embedded_vectors = []
    vector_store = []
    progress = st.progress(0, "Processing file...")
    
    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed_query(chunk)
        embedded_vectors.append(embedding)
        vector_store.append({
            "id": f"{file_hash}_{i}",
            "values": embedding,
            "metadata": {
                "file_name": file_name,
                "page_content": chunk,
                "page_number": i + 1
            }
        })
        progress.progress(int(((i + 1) / len(chunks)) * 90), f"Processing chunk {i + 1}/{len(chunks)}")

    index.upsert(vectors=vector_store, namespace=file_hash)
    progress.progress(100, "File processed successfully!")
    return {"namespace": file_hash, "texts": chunks, "embeddings": embedded_vectors}

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))

def get_conversation_chain():
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash")
    prompt = PromptTemplate(
        template="""You are a helpful and polite PDF chatbot. A user has uploaded a PDF document for reference.
Your behavior rules:
1. If the user sends a greeting only (e.g., 'hi', 'hello'): respond with a friendly greeting and invite PDF-based questions.
2. If there's a greeting + query: greet and answer if query is PDF-related.
3. If there's no greeting: respond based on whether query is PDF-related.
Never hallucinate. Be concise and relevant.

Context: {context}

Question: {question}
""",
        input_variables=["context", "question"]
    )
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

def handle_user_query(query):
    if not query:
        return "Please ask a question."
    store = st.session_state.vector_store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GOOGLE_API_KEY)
    query_embedding = embeddings.embed_query(query)

    similarity = sorted([
        (cosine_similarity(query_embedding, vec), text)
        for vec, text in zip(store["embeddings"], store["texts"])
    ], reverse=True)

    context = [text for _, text in similarity[:3]]
    if not context:
        return "No relevant information found in the document."

    docs = [Document(page_content=chunk) for chunk in context]
    chain = st.session_state.qa_chain
    return chain.run(input_documents=docs, question=query)

# Streamlit App UI
st.title("📄 AskMyPDF - AI PDF Chatbot")

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

if not st.session_state.file_uploaded:
    st.subheader("📤 Upload your PDF file")
    file = st.file_uploader("", type=["pdf"])
    if st.button("Upload"):
        if file:
            full_text = extract_text_from_pdf(file)
            chunks = split_text_chunks(full_text)
            vector_store = load_vector_data(chunks, file.name)
            st.session_state.vector_store = vector_store
            st.session_state.qa_chain = get_conversation_chain()
            st.session_state.file_uploaded = True
            st.rerun()
        else:
            st.warning("⚠️ Please upload a PDF file.")

if st.session_state.file_uploaded:
    st.success("✅ File uploaded successfully! Start chatting below.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    col1, col2 = st.columns([1, 7], vertical_alignment="bottom")
    with col1:
        if st.button("🔄 Refresh"):
            st.session_state.clear()
            st.rerun()
    with col2:
        if st.button("🧹 Clear Chat"):
            st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a question about the PDF:")
    if user_input:
        with st.spinner("Processing your query..."):
            response = handle_user_query(user_input)

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)











# import os
# import stat
# import hashlib
# import numpy as np
# import streamlit as st
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from langchain.schema import Document
# from pinecone import Pinecone

# # Load environment variables
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# # Validate API keys
# if not GOOGLE_API_KEY or not PINECONE_API_KEY:
#     st.error("API keys are missing. Please check your .env file.")
#     st.stop()

# # Initialize Pinecone
# pc = Pinecone(api_key=PINECONE_API_KEY)
# INDEX_NAME = "pdf-chatbot-index"
# index = pc.Index(INDEX_NAME)

# # Utility functions
# def remove_readonly(func, path, _):
#     os.chmod(path, stat.S_IWRITE)
#     func(path)

# def extract_text_from_pdf(file):
#     reader = PdfReader(file)
#     return "".join(page.extract_text() or "" for page in reader.pages)

# def split_text_chunks(text):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
#     return splitter.split_text(text)

# def load_vector_data(chunks, file_name):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GOOGLE_API_KEY)
#     file_hash = hashlib.sha256(file_name.encode()).hexdigest()
#     embedded_vectors = []
#     vector_store = []
#     progress = st.progress(0, "Processing file...")
    
#     for i, chunk in enumerate(chunks):
#         embedding = embeddings.embed_query(chunk)
#         embedded_vectors.append(embedding)
#         vector_store.append({
#             "id": f"{file_hash}_{i}",
#             "values": embedding,
#             "metadata": {
#                 "file_name": file_name,
#                 "page_content": chunk,
#                 "page_number": i + 1
#             }
#         })
#         progress.progress(int(((i + 1) / len(chunks)) * 90), f"Processing chunk {i + 1}/{len(chunks)}")

#     index.upsert(vectors=vector_store, namespace=file_hash)
#     progress.progress(100, "File processed successfully!")
#     return {"namespace": file_hash, "texts": chunks, "embeddings": embedded_vectors}

# def cosine_similarity(a, b):
#     a, b = np.array(a), np.array(b)
#     if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
#         return 0.0
#     return float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))

# def get_conversation_chain():
#     llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash")
#     prompt = PromptTemplate(
#         template="""You are a helpful and polite PDF chatbot. A user has uploaded a PDF document for reference.
# Your behavior rules:
# 1. If the user sends a greeting only (e.g., 'hi', 'hello'): respond with a friendly greeting and invite PDF-based questions.
# 2. If there's a greeting + query: greet and answer if query is PDF-related.
# 3. If there's no greeting: respond based on whether query is PDF-related.
# Never hallucinate. Be concise and relevant.

# Context: {context}

# Question: {question}""",
#         input_variables=["context", "question"]
#     )
#     return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

# def handle_user_query(query):
#     if not query:
#         return "Please ask a question."
#     store = st.session_state.vector_store
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GOOGLE_API_KEY)
#     query_embedding = embeddings.embed_query(query)

#     similarity = sorted([
#         (cosine_similarity(query_embedding, vec), text)
#         for vec, text in zip(store["embeddings"], store["texts"])
#     ], reverse=True)

#     context = [text for _, text in similarity[:3]]
#     if not context:
#         return "No relevant information found in the document."

#     docs = [Document(page_content=chunk) for chunk in context]
#     chain = st.session_state.qa_chain
#     return chain.run(input_documents=docs, question=query)

# # Streamlit App UI
# st.title("📄 AskMyPDF - AI PDF Chatbot")

# if "file_uploaded" not in st.session_state:
#     st.session_state.file_uploaded = False

# if not st.session_state.file_uploaded:
#     st.subheader("📤 Upload your PDF file")
#     file = st.file_uploader("", type=["pdf"])
#     if st.button("Upload"):
#         if file:
#             full_text = extract_text_from_pdf(file)
#             chunks = split_text_chunks(full_text)
#             vector_store = load_vector_data(chunks, file.name)
#             st.session_state.vector_store = vector_store
#             st.session_state.qa_chain = get_conversation_chain()
#             st.session_state.file_uploaded = True
#             st.rerun()
#         else:
#             st.warning("⚠️ Please upload a PDF file.")

# if st.session_state.file_uploaded:
#     st.success("✅ File uploaded successfully! Start chatting below.")

#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     col1, col2 = st.columns([1, 7], vertical_alignment="bottom")
#     with col1:
#         if st.button("🔄 Refresh"):
#             st.session_state.clear()
#             st.rerun()
#     with col2:
#         if st.button("🧹 Clear Chat"):
#             st.session_state.messages = []

#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     user_input = st.chat_input("Ask a question about the PDF:")
#     if user_input:
#         with st.spinner("Processing your query..."):
#             response = handle_user_query(user_input)

#         st.session_state.messages.append({"role": "user", "content": user_input})
#         with st.chat_message("user"):
#             st.markdown(user_input)

#         st.session_state.messages.append({"role": "assistant", "content": response})
#         with st.chat_message("assistant"):
#             st.markdown(response)
