import os
import time
from langchain_groq import ChatGroq  #type:ignore
from pathlib import Path
from dotenv import load_dotenv  #type:ignore
from langchain_core.prompts import PromptTemplate #type:ignore
from langchain_core.output_parsers import StrOutputParser #type:ignore


env_path=Path(__file__).parent.parent/".env"
load_dotenv(env_path)
api_key=os.getenv('GROQ_API_KEY')

if not api_key:
    print("Api_Key_Not_Found")
    exit()
    
else:
    print("Initalizing Model For You........")
    


model=ChatGroq(
    model='openai/gpt-oss-20b',
    api_key=api_key,
    temperature=1.2,
    max_tokens=1200
    )

time.sleep(2)

#1st Prompt ->Detailed topic

template1=PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

#2nd Prompt ->Summary
template2=PromptTemplate(
    template="Write a 5 Line summary on the following text./n {text}.",
    input_variables=['text']
)

parser=StrOutputParser()

chain= template1 | model | parser | template2 | model | parser

result=chain.invoke({'topic':'blackhole'})
print(result)

# python3 strouputparser1.py 