import os
import time
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint  #type:ignore
from pathlib import Path
from dotenv import load_dotenv  #type:ignore
from langchain_core.prompts import PromptTemplate #type:ignore
from langchain_core.output_parsers import StrOutputParser #type:ignore

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

prompt=PromptTemplate(
    template="Generate 5 intesrting facts about {topic}",
    input_variables=['topic']
)

parser=StrOutputParser()

chain= prompt | model | parser
result=chain.invoke({'topic':'earth'})
print(result)

chain.get_graph().print_ascii()

#  python3 simple_chain.py