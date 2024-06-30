import streamlit as st
import subprocess
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_google_community import GoogleDriveLoader
from dotenv import load_dotenv

DEFAULT_FOLDER_ID = "1D2lETD9nsFPIxw4GE3laO_SdBPu3dQNO"

# Run setup script to ensure credentials are in the correct location
subprocess.run(['sh', './setup.sh'], check=True)


class Main:
    def __init__(self):
        self.load_env_variables()
        self.retriever = None
        if 'retriever_initialized' not in st.session_state:
            self.initialize_retriever(DEFAULT_FOLDER_ID)

    def load_env_variables(self):
        """Loads environment variables from .env file."""
        load_dotenv('var.env')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
        self.google_credentials_path = os.getenv(
            "GOOGLE_CREDENTIALS_PATH", ".credentials/credentials.json")

    def initialize_retriever(self, folder_id):
        """Initializes the retriever with documents from the specified directory path."""
        loader = GoogleDriveLoader(
            folder_id=folder_id,
            credentials_path=self.google_credentials_path,
            recursive=False
        )
        documents = loader.load()
        st.write(f"Loaded {len(documents)} documents from Google Drive")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()

        Pinecone(api_key=self.pinecone_api_key, environment='us-east-1-aws')
        vectbd = PineconeVectorStore.from_documents(
            docs, embeddings, index_name=self.pinecone_index_name)
        self.retriever = vectbd.as_retriever()
        st.session_state.retriever_initialized = True
        st.session_state.retriever = self.retriever
        st.write("Retriever initialized successfully")

    def chat_drive(self, query):
        if self.retriever is None:
            st.error("Retriever is not initialized.")
            return None

        llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")
        chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=self.retriever)

        answer = chain.invoke(query)
        return answer['result']


def main():
    st.sidebar.title("Document QA System with Google Drive")
    st.sidebar.write(
        "Upload your documents to Google Drive and interact with them using AI.")

    m = Main()

    folder_id = st.sidebar.text_input("Enter Google Drive Folder ID:", "")
    if st.sidebar.button("Update Retriever"):
        m.initialize_retriever(folder_id)
        st.sidebar.success("Retriever Updated successfully!")

    st.title("Chat with your Documents")
    if st.session_state.retriever_initialized:
        query = st.text_input("Ask a question about your documents:", "")
        if st.button("Get Answer"):
            m.retriever = st.session_state.retriever  # Ensure retriever is assigned
            answer = m.chat_drive(query)
            if answer:
                st.write(f"AI Assistant: {answer}")
            st.write("*********************************")
    else:
        st.warning("Please update the retriever first.")


if __name__ == "__main__":
    main()
