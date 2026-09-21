import os
import time
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint  #type:ignore
from pathlib import Path
from dotenv import load_dotenv  #type:ignore
from langchain.output_parsers import StructuredOutputParser, ResponseSchema  #type:ignore
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
    repo_id="google/gemma-2-2b-it",
    task="text_generation",
    huggingfacehub_api_token=api_key
)


model=ChatHuggingFace(llm=llm)
time.sleep(2)


schema=[
    ResponseSchema(name='fact_1',description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2',description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3',description='Fact 3 about the topic')
]




parser=StructuredOutputParser.from_response_schemas(schema)

tempalate=PromptTemplate(
    template='Give 3 facts about the {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain= tempalate | model | parser

result=chain.invoke({'topic':'blackhole'})

print(result)

# python3 structureoutputparser.py 
