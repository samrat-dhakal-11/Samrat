import os
import time
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint  #type:ignore
from pathlib import Path
from dotenv import load_dotenv  #type:ignore
from langchain_core.prompts import PromptTemplate #type:ignore


env_path=Path(__file__).parent.parent/".env"
load_dotenv(env_path)
api_key=os.getenv('HUGGINGFACE_ACESS_TOKEN')

if not api_key:
    print("Api_Key_Not_Found")
    exit()
    
else:
    print("Initalizing Model For You........")
    
    
    
llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4.1-Flash",
    task="text_generation",
    huggingfacehub_api_token=api_key
)


model=ChatHuggingFace(llm=llm)
time.sleep(2)

#1st Prompt ->Detailed topic

template1=PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

#2nd Prompt ->Summary
template2=PromptTemplate(
    template="Write a 5 Line summary on the following text./n {text}",
    input_variables=['text']
)

prompt1=template1.invoke({'topic':'blackhole'})
result=model.invoke(prompt1)


prompt2=template2.invoke({'text':result.content})
result=model.invoke(prompt2)


print(result.content)


# python3 str_ouput_parser.py