import os
import time
from langchain_groq import ChatGroq #type:ignore
from pathlib import Path
from dotenv import load_dotenv  #type:ignore
from langchain_core.prompts import PromptTemplate #type:ignore
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser #type:ignore
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda #type:ignore
from pydantic import BaseModel,Field #type:ignore
from typing import Literal

env_path=Path(__file__).parent.parent/".env"
load_dotenv(env_path)
api_key=os.getenv('GROQ_API_KEY')


if not api_key :
    print("Api_Key_Not_Found")
    exit()
    
else:
    print("Initalizing Model For You........")
  
#______________Model Setup____________________  
 
 #__________________ Initalizing Close Source model __________________________  
    
model=ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=api_key
)

time.sleep(2)

#____________________________________________________________

parser=StrOutputParser()

class Feedback(BaseModel):
    sentiment:Literal['positive','negative']=Field(description="Gives the sentiment of the feedback")

parser2=PydanticOutputParser(pydantic_object=Feedback)


prompt1=PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction} ",
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain= prompt1 | model | parser2

prompt2=PromptTemplate(
    template="Write an apporpriate  response to this positive feedback \n {feedback}",
    input_variable=['feedback']
)

prompt3=PromptTemplate(
    template="Write an apporpriate  response to this neagtive feedback \n {feedback}",
    input_variable=['feedback']
)


branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='positive', prompt2 | model | parser),
    (lambda x:x.sentiment=='negative',prompt2 | model | parser),
    RunnableLambda(lambda x:"could not find sentiment ")
)

chain= classifier_chain | branch_chain

result=chain.invoke({'feedback':"This is a Wonderful phone"})

print(result)

chain.get_graph().print_ascii()
# python3 conditional_chain.py