import streamlit as st
from lida import Manager, TextGenerationConfig, llm
from lida.datamodel import Goal
import os
import pandas as pd
from langchain.callbacks import LLMonitorCallbackHandler
from PIL import Image
from io import BytesIO
import base64

# make data dir if it doesn't exist
os.makedirs("data", exist_ok=True)

st.set_page_config(
    page_title="WMC Viz Agents Playground :seedling:",
    page_icon="📊",
)

st.subheader("WMC DMI-Agents Playground :seedling:")


# Step 1 - Get OpenAI API key
openai_key = os.getenv("OPENAI_API_KEY")

selected_model = "gpt-3.5-turbo"
temperature = 0.2
use_cache = True

#st.sidebar.write("### Choose a dataset")

datasets = [
    {"label": "Select a dataset", "url": None},
    {"label": "Cars", "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv"},
    {"label": "Weather", "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/weather.json"},
]

selected_dataset_label = st.sidebar.selectbox(
    'Choose a dataset',
    options=[dataset["label"] for dataset in datasets],
    index=0
)

#upload_own_data = st.sidebar.checkbox("Select a dataset from Big Query")
selected_dataset = datasets[[dataset["label"]
                            for dataset in datasets].index(selected_dataset_label)]["url"]

if not selected_dataset:
    st.info("To continue, select a dataset from the sidebar on the left or upload your own.")
selected_method = 'default'

# Step 3 - Generate data summary
if openai_key and selected_dataset and selected_method:
    handler = LLMonitorCallbackHandler()
    lida = Manager(text_gen=llm("openai", api_key=openai_key))
    textgen_config = TextGenerationConfig(
        n=1,
        temperature=temperature,
        model=selected_model,
        use_cache=use_cache)
            
    st.write("#### Summary")
    # **** lida.summarize *****
    summary = lida.summarize(
        selected_dataset,
        summary_method=selected_method,
        textgen_config=textgen_config)

    if "dataset_description" in summary:
        st.write(summary["dataset_description"])

    if "fields" in summary:
        fields = summary["fields"]
        nfields = []
        for field in fields:
            flatted_fields = {}
            flatted_fields["column"] = field["column"]
            # flatted_fields["dtype"] = field["dtype"]
            for row in field["properties"].keys():
                if row != "samples":
                    flatted_fields[row] = field["properties"][row]
                else:
                    flatted_fields[row] = str(field["properties"][row])
            # flatted_fields = {**flatted_fields, **field["properties"]}
            nfields.append(flatted_fields)
        nfields_df = pd.DataFrame(nfields)
        st.write(nfields_df)
    else:
        st.write(str(summary))

    goals = lida.goals(summary, n=4, textgen_config=textgen_config)

    st.write ("#### Suggested Questions") 
    goal_questions = [goal.visualization for goal in goals]
    st.write(goal_questions)
    #goal_questions.append("----Ask my own question----")
    # selected_goal = st.selectbox('Choose a generated question or ask own', 
    #                     options=goal_questions, 
    #                     index=None,
    #                     placeholder="Choose one"
    #                     )
    st.text_input("Own Question", label_visibility="collapsed", placeholder= 'Paste a suggested question or ask your own', key = 'user_query')
    
    def recreate_graph():
        with container2:
            #st.write("method2 called")
            edit_query = st.session_state.edit_query
            st.info("Your Edits: " + edit_query)
            edited_charts = lida.edit(code=st.session_state.code,  summary=summary, instructions=edit_query, library='seaborn', textgen_config=textgen_config)
            #st.write(len(edited_charts))  
            edited_imgdata = base64.b64decode(edited_charts[0].raster)
            edited_img = Image.open(BytesIO(edited_imgdata))
            st.image(edited_img, use_column_width=True)
            with st.expander("See code"):
                st.code(edited_charts[0].code)
       
    def generate_graph():
        with container1:
            #st.write("method called")
            user_query = st.session_state.user_query
            st.info("Your Query: " + user_query)
            charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config, library='seaborn')  
            #st.write(len(charts))
            imgdata = base64.b64decode(charts[0].raster)
            img = Image.open(BytesIO(imgdata))
            st.image(img, use_column_width=True)
            with st.expander("See code"):
                st.code(charts[0].code)
            st.text_input("Enter instructions to modify chart", key = 'edit_query')
            st.session_state.code = charts[0].code
            st.button("Re-Create Graph", on_click=recreate_graph)
            
    
    st.button("Generate Graph", on_click=generate_graph)
    container1 = st.container()
    container1.empty()
    container2 = st.container()
    container2.empty()
    
