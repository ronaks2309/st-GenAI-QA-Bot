# Create virtual Env >> python -m venv .yourname_venv
# Command to activate Virt Env >> yourname_venv/Scripts/activate
# Command to install requirements >> pip install -r requirements.txt
# Command to exit the venv >> deactivate
# Command to launch >> streamlit run app.py

import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os 
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone  
import pinecone

#### PREPARATION #### 
def create_vector_index_from_pdf(uploaded_files,password):
    print("get_vector_index_from_pdf called")
    if password != st.secrets.APP_PASSWORD: 
        st.warning("Incorrect Password")
        return
    if check_index() != "Valid":
        st.warning("Index Name Not Valid")
        return
    data_folder = save_files(uploaded_files)
    text_chunks = get_chunks(data_folder)
    vector_db = get_vector_db(text_chunks)
    return vector_db

def save_files(uploaded_files):
    # delete old files
    for filename in os.listdir("data"):
        if os.path.isfile(os.path.join('data', filename)):
            os.remove(os.path.join('data', filename))
    # save files by looping over the uploaded files
    for file in uploaded_files:
        with open(os.path.join(os.getcwd(),"data", file.name), "wb") as f:
            f.write(file.getbuffer())
        st.success("Saved File to Data: "+file.name)
    return("data")

def get_chunks(data_folder):
    print("get chunks called")
    loader = DirectoryLoader(os.path.join(os.getcwd(),data_folder), loader_cls = PyPDFLoader)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(pages)
    return text_chunks

def get_vector_db(text_chunks):
    print("get vector db called")
    embeddings = OpenAIEmbeddings()
    pinecone.init(environment='gcp-starter')
    new_index = st.session_state.new_index
    pinecone.create_index(name=new_index, metric="cosine", dimension=1536)
    pinecone_db = Pinecone.from_documents(text_chunks, embeddings, index_name=new_index)
    return pinecone_db

def check_index():
    pinecone.init(environment='gcp-starter')
    index_name = st.session_state.new_index
    if index_name == "":
        st.warning('Index name cannot be blank', icon="⚠️")
        return "Invalid"
    elif index_name in pinecone.list_indexes():
        st.warning('Index already exists', icon="⚠️")
        return "Invalid"
    else: 
        st.success('Index name available :white_check_mark:')
        return "Valid"

def main():
    print("Main method called")
    load_dotenv()
    st.set_page_config(page_title="DB Creator", page_icon = ":books:")
    sideb = st.sidebar
    #st.title("Create a vector index from knowledge files :database:")
    st.subheader("Load PDF files into a new Pinecone Index")
    
    col1, col2 = st.columns(2)
    col1.text_input("New Index Name", label_visibility='collapsed', placeholder="Enter name for New Pinecone Index", key='new_index')
    col2.button("Check", on_click= check_index)
    
    uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)
    
    st.text_input("Enter Password", type = "password", placeholder="Enter Password", key='password')    
    
    if st.button("Create"):
        with st.spinner("Processing"):
            pinecone_index = create_vector_index_from_pdf(uploaded_files,st.session_state.password)
            st.success("Pinecone Index created successfully !!")
    
    



if __name__ == '__main__':
    main()


