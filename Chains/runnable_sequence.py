import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from contextlib import redirect_stdout, redirect_stderr


env_path=Path(__file__).parent.parent /".env"
load_dotenv(env_path)

api_key=os.getenv("GEMINI_API_KEY")

if not api_key:
    print("API KEY MISSING")
    exit()
    
else:
    print("Initializing Model.....")

parser=StrOutputParser()

model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',api_key=api_key)

prompt1=PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template="Explain the following joke - {text} \n Output shouldnot contain  ** and ###",
    input_variables=['text']
)

chain=RunnableSequence(prompt1,model,parser,prompt2,model,parser)

time.sleep(2)
with open(os.devnull, 'w') as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
    result = chain.invoke({'topic':'AI'})

print(result)


# python3 runnable_sequence.py