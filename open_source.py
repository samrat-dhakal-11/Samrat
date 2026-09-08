import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st

load_dotenv()
api_key=os.getenv('HUGGINGFACEHUB_API_TOKEN')

model=HuggingFaceEndpointEmbeddings(
    model='Qwen/Qwen3-Embedding-8B'
    )


#                        streamlit run open_source.py

st.title("SAMRATTTT")
st.header("Open Source Embdeeing Example:")

options = [
    "Volcanoes create new land when lava cools down.",
    "Honey never spoils even after thousands of years.",
    "Bicycles are the most energy-efficient human transportation mode.",
    "Sound travels four times faster underwater than air.",
    "Rainforests produce over twenty percent of world oxygen."
]

st.selectbox("Look at the diffrent options here",options)
st.divider()

user_choose=st.text_input("Write any one unique word form the any five options ")
st.divider()

if st.button("Submit"):

    embedd_options = model.embed_documents(options)
    embedd_query = model.embed_query(user_choose)

    with st.expander("See Embedding of both Option and USer Query"):
     st.write(f"Option Embeddings {embedd_options}")
     st.divider()
     st.write(f"Query Embedding {embedd_query}")

    st.write("Now finding the cosine similarity btn these and getting the ans")
    
    
    
    s=cosine_similarity([embedd_query],embedd_options)
    st.write(s)
    
    
    similarity=np.argmax(cosine_similarity([embedd_query],embedd_options)[0])
    st.write(f"Index having high cosine similarity with user query in document is {similarity}")
    
    st.write(f"User query:{user_choose}")
    st.write(f" Required Option: {options[similarity]}")



st.divider()

with st.expander("Additional Info:"):
    info="This is made by Samrat Dhakal"
    github="Github Profile:   github.com/samrat-dhakal-11/"
    linkedin="Linkedin :  linkedin.com/in/samrat-dhakal/b323a9416/"
    
    st.code(info)
    st.code(github)
    st.code(linkedin)
    
st.divider()


code_to_display="""   
import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st

load_dotenv()
api_key=os.getenv('HUGGINGFACEHUB_API_TOKEN')

model=HuggingFaceEndpointEmbeddings(
    model='Qwen/Qwen3-Embedding-8B'
    )


#                        streamlit run open_source.py

st.title("SAMRATTTT")
st.header("Open Source Embdeeing Example:")

options = [
    "Volcanoes create new land when lava cools down.",
    "Honey never spoils even after thousands of years.",
    "Bicycles are the most energy-efficient human transportation mode.",
    "Sound travels four times faster underwater than air.",
    "Rainforests produce over twenty percent of world oxygen."
]

st.selectbox("Look at the diffrent options here",options)
st.divider()

user_choose=st.text_input("Write any one unique word form the any five options ")
st.divider()

if st.button("Submit"):

    embedd_options = model.embed_documents(options)
    embedd_query = model.embed_query(user_choose)

    with st.expander("See Embedding of both Option and USer Query"):
     st.write(f"Option Embeddings {embedd_options}")
     st.divider()
     st.write(f"Query Embedding {embedd_query}")

    st.write("Now finding the cosine similarity btn these and getting the ans")
    
    
    
    s=cosine_similarity([embedd_query],embedd_options)
    st.write(s)
    
    
    similarity=np.argmax(cosine_similarity([embedd_query],embedd_options)[0])
    st.write(f"Index having high cosine similarity with user query in document is {similarity}")
    
    st.write(f"User query:{user_choose}")
    st.write(f" Required Option: {options[similarity]}")



st.divider()

with st.expander("Additional Info:"):
    info="This is made by Samrat Dhakal"
    github="Github Profile:   github.com/samrat-dhakal-11/"
    linkedin="Linkedin :  linkedin.com/in/samrat-dhakal/b323a9416/"
    
    st.code(info)
    st.code(github)
    st.code(linkedin)
    
st.divider()

with st.expander("Tap here to see this code"):
    code_to_display=.................
    st.code(code_to_display,language="Python")

"""  
with st.expander("Tap here to see this code"):
    st.code(code_to_display,language="Python")
