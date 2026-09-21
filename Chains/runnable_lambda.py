import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda


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
    template="Generate a joke on  {topic} \n Output shouldnot contain  ** and ###",
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template="Generate a Suitable Explanation  about {text} \n Output shouldnot contain  ** and ###",
    input_variables=['text']
)

def word_counter(text):
    return len(text.split())

runnable_lambda=RunnableLambda(word_counter)


joke_gen_chain=RunnableSequence(prompt1,model,parser)

parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'length':runnable_lambda,
    'explanation':RunnableSequence(prompt2,model,parser)
})

final_chain=RunnableSequence(joke_gen_chain,parallel_chain)

time.sleep(2)

result = final_chain.invoke({'topic':'AI'})

print(f"\n\n{result}\n\n")
print("-"*150)

print(f'\n\n\nJoke\n{result['joke']}\n\n')
print("-"*150)

print(f'\n\n\nWord Count\n{result['length']}\n\n')

print("-"*150)
print("\nExplanation\n",result['explanation'])

# or 

final_result="""\n\n\n\nJoke={} \n\nWord Count={}\n\nExplanation={}""".format(result['joke'],result['length'],result['explanation'])
print(f"{final_result}\n\n")

final_chain.get_graph().print_ascii()


# python3 runnable_lambda.py