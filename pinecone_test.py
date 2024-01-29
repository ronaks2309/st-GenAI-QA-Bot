import streamlit
import os
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="0cdf1bbc-c528-4859-b96d-1a1c3583b007")

# pc.create_index(
#     name="quickstart3",
#     dimension=8,
#     metric="euclidean",
#     spec=ServerlessSpec(
#         cloud='aws', 
#         region='us-west-2'
#     ) 
# ) 

active_indexes = pc.list_indexes()
print(active_indexes)
