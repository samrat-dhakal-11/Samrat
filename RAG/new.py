import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnablePassthrough,RunnableBranch


env_path=Path(__file__).parent.parent /".env"
load_dotenv(env_path)

api_key=os.getenv("GEMINI_API_KEY")


if not api_key:
    print("API KEY MISSING")
    exit()
    
else:
    print("Initializing Model.....")

parser=StrOutputParser()

model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',api_key=api_key,temperature=1.5)
