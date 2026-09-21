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

prompt1=PromptTemplate(
    template="Write a detail report on {topic} \n Output shouldnot contain  ** and ###",
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template="Summarize the following text \n {text} \n Output shouldnot contain  ** and ###",
    input_variables=['text']
)

report_gen_chain=RunnableSequence(prompt1,model,parser)

branch_chain=RunnableBranch(
    (lambda x : len(x.split()) >500,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain=RunnableSequence(report_gen_chain,branch_chain)

time.sleep(2)

result = final_chain.invoke({'topic':'Russia VS Ukraine'})

print(f"\n\n{result}\n\n")

# python3 runnable_branch.py