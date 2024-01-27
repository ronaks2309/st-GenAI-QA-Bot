# Create virtual Env >> python -m venv .yourname_venv
# Command to activate Virt Env >> .stgenai_venv/Scripts/activate
# Command to install requirements >> pip install -r requirements.txt
# Command to exit the venv >> deactivate
# Command to launch >> streamlit run streamlit_app.py

import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os 
#from langchain.document_loaders import DirectoryLoader, PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import LLMonitorCallbackHandler
import pinecone
from langchain.vectorstores import Pinecone  


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]



def get_conversation_chain(selected_index):
    print("get conversation chain called")
    #check for password
    if st.session_state.password != st.secrets.APP_PASSWORD:
        st.warning("Incorrect Password")
        return
    handler = LLMonitorCallbackHandler()
    embeddings = OpenAIEmbeddings()
    pinecone.init(environment='gcp-starter')
    index_name = selected_index
    vector_db = Pinecone.from_existing_index(index_name, embeddings)
    llm = ChatOpenAI(model = 'gpt-3.5-turbo-1106', callbacks=[handler])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state.retriever=vector_db.as_retriever(search_kwargs={"k": 3})
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type='stuff', 
        retriever=st.session_state.retriever,
        memory=memory, 
        callbacks=[handler],
        #return_source_documents = True
        )
    st.success("Pinecone index selected: "+index_name)
    return conversation_chain



#### USER INQUIRY ####
def display_chats():
    print("display chats method called")
    reply_container = st.container()
    container = st.container()
    
    with container:
        st.text_input("Question:", placeholder="Ask about your PDF", key='input', on_change = submit)


    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji") 
    
def submit():
    print ("Submit method called")
    st.session_state.user_question = st.session_state.input
    st.session_state.input = ""
    with st.spinner('Generating Response...'):
        handle_user_question(st.session_state['user_question'])
    return

def handle_user_question(user_question):
    print("handle user question called")
    if 'chain' not in st.session_state:
        st.warning("No Agent is selected. Please select agent, enter password and press go")
        return

    mychain = st.session_state.chain
    result = mychain({"question": user_question, "chat_history": st.session_state['history']})
    st.session_state['history'].append((user_question, result["answer"]))
    src_docs = st.session_state.retriever.get_relevant_documents(user_question)
    unique_ref_text = get_unique_references(src_docs)
    st.session_state['past'].append(user_question)
    st.session_state['generated'].append(result["answer"] + "\n\n" + "To Learn more, visit: \n" + unique_ref_text) 
    return

def get_unique_references(src_docs):
    # create list of sources
    src_list = []
    for doc in src_docs:
        src_list.append(doc.metadata.get('source').rpartition("\\")[-1])
    #deduplicate
    unique_src_list = list(dict.fromkeys(src_list))
    unique_ref_text = "\n".join(unique_src_list)
    return unique_ref_text


def main():
    print("Main method called")
    load_dotenv()
    initialize_session_state()
    #get_conversation_chain('wmc-faq')
    st.set_page_config(page_title="WMC FAQs with LLM", page_icon = ":books:")
    st.title("WMC FAQs with LLMs :books:")
    sideb = st.sidebar
    #st.sidebar.title("Select Pinecone Index")
    selected_index = sideb.selectbox(
        "Select an agent", 
        ["wmc-faq", "dummy-1", "dummy-2"],
        index=None,
        placeholder="Choose wmc-faq and click Go")
    sideb.text_input("Password", type = "password", placeholder="Enter Password", key='password')
    if st.sidebar.button("Go"):
        with st.spinner("Connecting to vector dB"):
            st.session_state.chain = get_conversation_chain(selected_index)
    if st.sidebar.button("Clear History"):
        st.session_state['history'] = []
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]
        st.session_state['past'] = ["Hey! 👋"]
    
    display_chats()


if __name__ == '__main__':
    main()

