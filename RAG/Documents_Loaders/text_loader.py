import os
import time
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


file_path = "/Users/pratiksha/Documents/PYTHON__XHIS_XHIS/Gen_Ai/nwee/helloo_there/samrat/Samrat/RAG/Documents_Loaders/cricket.txt"
env_path=Path(__file__).parent.parent.parent /".env"
load_dotenv(env_path)

api_key=os.getenv("GEMINI_API_KEY")


if not api_key:
    print("API KEY MISSING")
    exit()
    
else:
    print("Initializing Model.....")


model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',api_key=api_key,temperature=1.5)

prompt=PromptTemplate(
    template="Write a summary for the following poem - \n{poem} and the output shouldnot have ** and ### and make summary in 1 paragraph",
    input_variables=['poem']
)
parser=StrOutputParser()

loader=TextLoader(file_path,encoding='utf-8')
docs=loader.load()
time.sleep(2)

print(docs[0].page_content)
print(docs[0].metadata)

chain=prompt | model | parser

result=chain.invoke({'poem':docs[0].page_content})
print("\n\n\n\n")
print(result)