import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel


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
    template="Generate a tweet about {topic} \n Output shouldnot contain  ** and ###",
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template="Generate a Linkedin post  about {topic} \n Output shouldnot contain  ** and ###",
    input_variables=['topic']
)

parallel_chain=RunnableParallel({
    'tweet':RunnableSequence(prompt1,model,parser),
    'linkedin':RunnableSequence(prompt2,model,parser)
})


time.sleep(2)

result = parallel_chain.invoke({'topic':'AI'})

print(f'\n\n\nTweet\n{result['tweet']}\n\n')
print("-"*150)
print("\nLinkedin\n",result['linkedin'])


# python3 runnable_parallel.py