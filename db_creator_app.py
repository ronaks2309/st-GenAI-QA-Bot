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
from pinecone import Pinecone as pinecone_Pinecone
from pinecone import ServerlessSpec

#### PREPARATION #### 
def create_vector_index_from_pdf(uploaded_files,password):
    print("get_vector_index_from_pdf called")
    if password != st.secrets.APP_PASSWORD: 
        st.warning("Incorrect Password")
        return
    if check_index() != "Valid":
        st.warning("Agent Name Not Valid")
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
    pc = pinecone_Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    new_index = st.session_state.new_index
    pc.create_index(name=new_index, metric="cosine", dimension=1536,
                    spec=ServerlessSpec(cloud='aws', region='us-west-2') 
                    )
    pinecone_db = Pinecone.from_documents(text_chunks, embeddings, index_name=new_index)
    return pinecone_db

def check_index():
    pc = pinecone_Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index_name = st.session_state.new_index
    if index_name == "":
        st.warning('Agent name cannot be blank', icon="⚠️")
        return "Invalid"
    elif index_name in pc.list_indexes():
        st.warning('Agent already exists', icon="⚠️")
        return "Invalid"
    else: 
        st.success('Agent name available :white_check_mark:')
        return "Valid"

def main():
    print("Main method called")
    load_dotenv()
    st.set_page_config(page_title="Custom Agents Builder", page_icon = ":books:")
    sideb = st.sidebar
    #st.title("Create a vector index from knowledge files :database:")
    st.subheader("Build your Agent with Domain Specific Knowledge")
    
    col1, col2 = st.columns(2)
    col1.text_input("Agent Name", label_visibility='collapsed', placeholder="Enter name for New Agent", key='new_index')
    col2.button("Check", on_click= check_index)
    
    st.text_area("Agent Description", placeholder="Enter a short description of your agent")
    
    st.text_area("Agent Instructions",placeholder="Enter Special instructions that you want the agent to follow.\n For ex. Always be courteous. You replies should only be based on the domain knowledge provided. If you don't know the answer, guide the user to speak with a WMC Associate. Do not give financial advice.")

    uploaded_files = st.file_uploader("Upload Domain Knowledge files", accept_multiple_files=True)
    
    st.text_input("Enter Password", type = "password", placeholder="Enter Password", key='password')    
    
    if st.button("Create"):
        with st.spinner("Processing"):
            pinecone_index = create_vector_index_from_pdf(uploaded_files,st.session_state.password)
            st.success("Agent created successfully !!")
    
    



if __name__ == '__main__':
    main()


