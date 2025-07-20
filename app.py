## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## Set up Streamlit 
st.title("Conversational RAG With PDF uploads and chat history")
st.write("Upload PDFs and chat with their content")

## Input the Groq API Key
api_key = st.text_input("Enter your Groq API key:", type="password")

## Check if Groq API key is provided
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    ## Chat interface
    session_id = st.text_input("Session ID", value="default_session")

    # Stateful chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}

    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    uploaded_files = st.file_uploader("Choose PDF file(s)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_pdf = "./temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

        # Split and embed documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = FAISS.from_documents(splits, embedding=embeddings)
        else:
            new_vs = FAISS.from_documents(splits, embedding=embeddings)
            st.session_state.vectorstore.merge_from(new_vs)

        retriever = st.session_state.vectorstore.as_retriever()

        # Prompts
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question, which might reference context in the chat history, "
                       "formulate a standalone question. Do NOT answer the question, just reformulate it."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # Chain creation
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # User Input
        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)

else:
    st.warning("Please enter your Groq API Key to proceed.")
