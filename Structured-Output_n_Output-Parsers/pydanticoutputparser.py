import os
import time
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint  #type:ignore
from pathlib import Path
from dotenv import load_dotenv  #type:ignore
from langchain_core.prompts import PromptTemplate #type:ignore
from langchain_core.output_parsers import PydanticOutputParser #type:ignore
from pydantic import BaseModel,Field #type:ignore

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

class Person(BaseModel):
    name:str=Field(description="Name of the person")
    age:int=Field(gt=18,description="Age of the person")
    city:str=Field(description="Name of city the person belongs to")
    
parser=PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(
    template="Generate the name,age an city of a fictional {place} person \n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)


chain= template | model | parser

result=chain.invoke({'place':'nepal'})
print(result)


#  python3 pydanticoutputparser.py  