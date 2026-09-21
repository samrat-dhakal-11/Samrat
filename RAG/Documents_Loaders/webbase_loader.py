import os
import time
from dotenv import load_dotenv
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader

url="https://en.wikipedia.org/wiki/Artificial_intelligence"
loader=WebBaseLoader(url)
docs=loader.load()

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
    template="Answer the following questions \n {question} from the following text - \n{text} and the output shouldnot have ** and ###",
    input_variables=['question','text']
)
parser=StrOutputParser()
chain=prompt | model | parser
result=chain.invoke({'question':"Why is this popular nowadays?",'text':docs[0].page_content})
print(result)