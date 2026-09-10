import os
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

env_path=Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
api_key= os.getenv('GROQ_API_KEY')  

modell=ChatGroq(
  model="openai/gpt-oss-20b",
  api_key=api_key
)

messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about langchain")
]

result=modell.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)