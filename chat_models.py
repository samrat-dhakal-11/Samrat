import os
from langchain_groq import ChatGroq   
from dotenv import load_dotenv     
import streamlit as st               
load_dotenv()
api_key=os.getenv('GROQ_API_KEY')

model=ChatGroq(
    api_key=api_key,
    model='openai/gpt-oss-20b',
    temperature=0.7,
    max_tokens=1000
    )


st.title("Ask TO Samrat")
user_query=st.text_input("Whats Your Question")
if st.button("Ask"):
    result=model.invoke(user_query)
    st.write(result.content)

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
from langchain_groq import ChatGroq   
from dotenv import load_dotenv     
import streamlit as st               
load_dotenv()
api_key=os.getenv('GROQ_API_KEY')

model=ChatGroq(
    api_key=api_key,
    model='openai/gpt-oss-20b',
    temperature=0.7,
    max_tokens=1000
    )


st.title("Ask TO Samrat")
user_query=st.text_input("Whats Your Question")
if st.button("Ask"):
    result=model.invoke(user_query)
    st.write(result.content)

st.divider()

with st.expander("Additional Info:"):
    info="This is made by Samrat Dhakal"
    github="Github Profile:   github.com/samrat-dhakal-11/"
    linkedin="Linkedin :  linkedin.com/in/samrat-dhakal/b323a9416/"
    
    st.code(info)
    st.code(github)
    st.code(linkedin)
    
st.divider()

code_to_display=.............

with st.expander("Tap here to see this code"):
    st.code(code_to_display,language="Python")


"""


with st.expander("Tap here to see this code"):
    st.code(code_to_display,language="Python")

