import os  # To interact with environment variables and filesystem
import hashlib  # Not used in this snippet, but usually used to hash data (e.g., for caching)
from tempfile import NamedTemporaryFile  # Creates temporary files for handling uploads
import streamlit as st  # Streamlit for web UI
from dotenv import load_dotenv  # To load environment variables from .env files


# Import LangChain tools and integrations
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma  # Vector store for document embeddings
from langchain_community.chat_message_histories import ChatMessageHistory  # Keeps track of per-session chat history
from langchain_core.chat_history import BaseChatMessageHistory  # Abstract base for chat history
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Tools to define prompt templates
from langchain_core.runnables.history import RunnableWithMessageHistory  # Wraps RAG chains with chat history
from langchain_groq import ChatGroq  # Groq LLM integration
from langchain_huggingface import HuggingFaceEmbeddings  # For text embeddings using HuggingFace
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Splits PDFs into manageable chunks
from langchain_community.document_loaders import PyPDFLoader  # Loads and parses PDF files


# Load environment variables from a .env file (e.g., API keys)
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")  # Set Hugging Face token into environment


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="PDF Chat with RAG", page_icon="📄", layout="wide")
st.title("📄 Chat with your PDFs using RAG")
st.caption("Powered by LangChain, Chroma, HuggingFace, and Groq")

# ---------------- API Key & Session ----------------
api_key = st.text_input("🔑 Enter your Groq API key:", type="password")
session_id = st.text_input("💬 Session ID", value="default")

# Create a place to store chat history in session state (Streamlit's runtime memory)
if "store" not in st.session_state:
    st.session_state.store = {}

# ---------------- Embeddings  model loader with caching----------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------- LLM Selection ----------------
model_name = st.selectbox("Choose LLM model", ["llama3-8b-8192", "llama3-70b-8192"])

# LLM instantiation using Groq with caching
@st.cache_resource(show_spinner=False)
def get_llm(api_key: str, model_name: str):
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        streaming=True,  # Enables token streaming
    )

# Only initialize LLM if API key is provided
# Stop execution if API key is missing
if not api_key:
    st.warning("Please enter your Groq API key to start.")
    st.stop()

# Load and show LLM status
llm = get_llm(api_key, model_name)
st.success(f"✅ Loaded model: {model_name}")

# ---------------- Builds retriever from uploaded PDFs ----------------
@st.cache_resource(show_spinner=True)
def build_retriever(files):
    docs = [] # Stores parsed documents
     # For each uploaded PDF file
    for uf in files:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uf.getbuffer())  # Save file to disk temporarily
            tmp.flush()
            loader = PyPDFLoader(tmp.name)  # Load PDF with PyPDF
            docs.extend(loader.load())  # Extract text
    # Split text into chunks for embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    
    # Create Chroma vector store
    vectorstore = Chroma.from_documents(splits, embedding=get_embeddings())
    # Return a retriever that returns top 3 relevant chunks
    return vectorstore.as_retriever(search_kwargs={"k": 3})
# Returns chat history for a given session
def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

# ---------------- Main App Logic Upload UI for PDF files
uploaded_files = st.file_uploader("📎 Upload one or more PDF files", type="pdf", accept_multiple_files=True)

# Run only when files are uploaded
if uploaded_files:
    retriever = build_retriever(uploaded_files)  # Create retriever from files


    # Prompts to rephrase user query using history
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, rewrite it as a standalone question if necessary."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
     # Prompt to answer user question using retrieved documents
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the retrieved context to answer the question. If unsure, say 'I don't know'.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # RAG Chain
        # Create history-aware retriever (rephrases based on chat history)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    # QA chain that injects retrieved documents into LLM
    qa_chain = create_stuff_documents_chain(llm, answer_prompt)

    # Combine both into a retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # Wrap the chain with chat history for memory and context
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Chat input and streaming response
    # Chat input field
    user_q = st.chat_input("Ask something about the uploaded PDFs...")

    # Handle user query
    if user_q:
        history = get_session_history(session_id)  # Get session history
        placeholder = st.empty()  # Placeholder for streaming output

        with placeholder.container():
            response_text = ""
            # Stream token-by-token output
            for chunk in conversational_rag_chain.stream(
                {"input": user_q},
                config={"configurable": {"session_id": session_id}},
            ):
                delta = chunk.get("answer", "") or chunk.get("output", "")
                response_text += delta
                placeholder.markdown(response_text)  # Update response in real-time

        # Show full chat history after response
        st.divider()
        st.subheader("🕓 Chat History")
        for m in history.messages:
            role = "🧑‍💬" if m.type == "human" else "🤖"
            st.markdown(f"{role} **{m.content}**")
else:
    # Prompt user to upload files if not done yet
    st.info("Upload at least one PDF file to begin chatting.")
