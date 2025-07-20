## RAG Q&A Conversation With PDF Including Chat History - Cloud Ready
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile
import shutil

# Load environment variables (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available in cloud deployment

# Configuration - Use environment variables or Streamlit secrets
def get_api_keys():
    """Get API keys from environment variables or Streamlit secrets"""
    groq_api_key = None
    hf_token = None
    
    # Try to get from Streamlit secrets first (for cloud deployment)
    if hasattr(st, 'secrets'):
        try:
            groq_api_key = st.secrets.get("GROQ_API_KEY")
            hf_token = st.secrets.get("HF_TOKEN")
        except:
            pass
    
    # Fallback to environment variables
    if not groq_api_key:
        groq_api_key = os.getenv("GROQ_API_KEY")
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN")
    
    return groq_api_key, hf_token

# Get API keys
GROQ_API_KEY, HF_TOKEN = get_api_keys()

# Set HuggingFace token if available
if HF_TOKEN:
    os.environ['HF_TOKEN'] = HF_TOKEN

# Initialize embeddings
@st.cache_resource
def load_embeddings():
    """Load embeddings model (cached for performance)"""
    try:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error loading embeddings: {str(e)}")
        return None

embeddings = load_embeddings()

# Streamlit UI
st.set_page_config(
    page_title="Conversational RAG with PDFs",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Conversational RAG with PDF Uploads")
st.write("Upload PDFs and have intelligent conversations with their content using chat history!")

# API Key Status
col1, col2 = st.columns(2)
with col1:
    if GROQ_API_KEY:
        st.success("✅ Groq API Key: Configured")
    else:
        st.error("❌ Groq API Key: Not configured")

with col2:
    if HF_TOKEN:
        st.success("✅ HuggingFace Token: Configured")
    else:
        st.warning("⚠️ HuggingFace Token: Not configured (may affect performance)")

# Check if required API keys are available
if not GROQ_API_KEY:
    st.error("""
    **Missing Groq API Key!**
    
    For cloud deployment, add your API keys to Streamlit secrets:
    1. Go to your Streamlit Cloud dashboard
    2. Click on your app settings
    3. Go to the "Secrets" tab
    4. Add the following:
    ```
    GROQ_API_KEY = "your_groq_api_key_here"
    HF_TOKEN = "your_huggingface_token_here"
    ```
    
    For local development, create a `.env` file with:
    ```
    GROQ_API_KEY=your_groq_api_key_here
    HF_TOKEN=your_huggingface_token_here
    ```
    """)
    st.stop()

if not embeddings:
    st.error("Failed to load embeddings model. Please check your configuration.")
    st.stop()

# Initialize LLM
@st.cache_resource
def load_llm():
    """Load LLM (cached for performance)"""
    try:
        return ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Gemma2-9b-It")
    except Exception as e:
        st.error(f"Error loading LLM: {str(e)}")
        return None

llm = load_llm()

if not llm:
    st.error("Failed to initialize language model. Please check your Groq API key.")
    st.stop()

# Session management
st.sidebar.header("Session Configuration")
session_id = st.sidebar.text_input("Session ID", value="default_session", help="Use different session IDs to maintain separate conversation histories")

# Initialize session state
if 'store' not in st.session_state:
    st.session_state.store = {}
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

# File upload
st.header("📁 Upload Documents")
uploaded_files = st.file_uploader(
    "Choose PDF files", 
    type="pdf", 
    accept_multiple_files=True,
    help="Upload one or more PDF files to create a knowledge base"
)

# Process uploaded PDFs
if uploaded_files:
    # Check if we have new files
    current_files = {f.name for f in uploaded_files}
    new_files = current_files - st.session_state.processed_files
    
    if new_files or st.session_state.vectorstore is None:
        with st.spinner("Processing PDFs... This may take a moment."):
            try:
                documents = []
                
                for uploaded_file in uploaded_files:
                    if uploaded_file.name in new_files or st.session_state.vectorstore is None:
                        # Create a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_path = temp_file.name
                        
                        try:
                            # Load PDF
                            loader = PyPDFLoader(temp_path)
                            docs = loader.load()
                            
                            # Add source information
                            for doc in docs:
                                doc.metadata['source'] = uploaded_file.name
                            
                            documents.extend(docs)
                            
                        finally:
                            # Clean up temporary file
                            os.unlink(temp_path)
                
                if documents:
                    # Split documents
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=5000, 
                        chunk_overlap=500
                    )
                    splits = text_splitter.split_documents(documents)
                    
                    # Create or update vectorstore
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = Chroma.from_documents(
                            documents=splits, 
                            embedding=embeddings
                        )
                    else:
                        # Add new documents to existing vectorstore
                        st.session_state.vectorstore.add_documents(splits)
                    
                    st.session_state.processed_files = current_files
                    st.success(f"Successfully processed {len(uploaded_files)} PDF file(s)!")
                    
            except Exception as e:
                st.error(f"Error processing PDFs: {str(e)}")

# Chat interface
if st.session_state.vectorstore is not None:
    st.header("💬 Chat Interface")
    
    # Create RAG chain
    retriever = st.session_state.vectorstore.as_retriever()
    
    # Contextualize question prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # QA prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create chains
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    # Chat input and response
    user_input = st.text_input("Ask a question about your documents:", key="user_question")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Send", type="primary"):
            if user_input:
                with st.spinner("Thinking..."):
                    try:
                        response = conversational_rag_chain.invoke(
                            {"input": user_input},
                            config={"configurable": {"session_id": session_id}}
                        )
                        
                        st.success("**Assistant:** " + response['answer'])
                        
                        # Clear input by rerunning
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
    
    with col2:
        if st.button("Clear Chat History"):
            if session_id in st.session_state.store:
                del st.session_state.store[session_id]
            st.success("Chat history cleared!")
            st.rerun()
    
    # Display chat history
    if session_id in st.session_state.store:
        session_history = st.session_state.store[session_id]
        if session_history.messages:
            st.header("📋 Chat History")
            for i, message in enumerate(session_history.messages):
                if message.type == "human":
                    st.write(f"**You:** {message.content}")
                else:
                    st.write(f"**Assistant:** {message.content}")
                if i < len(session_history.messages) - 1:
                    st.divider()

else:
    st.info("👆 Please upload PDF files to start chatting!")

# Sidebar information
st.sidebar.header("ℹ️ Information")
st.sidebar.info("""
**How to use:**
1. Upload one or more PDF files
2. Wait for processing to complete
3. Ask questions about the content
4. The system remembers conversation history

**Features:**
- Multiple PDF support
- Conversation memory
- Session management
- Cloud deployment ready
""")

st.sidebar.header("📊 Session Status")
if st.session_state.vectorstore:
    st.sidebar.success("✅ Documents loaded")
else:
    st.sidebar.warning("⚠️ No documents loaded")

if session_id in st.session_state.store:
    msg_count = len(st.session_state.store[session_id].messages)
    st.sidebar.info(f"Messages in session: {msg_count}")
else:
    st.sidebar.info("Messages in session: 0")