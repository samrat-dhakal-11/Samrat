import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from pathlib import Path

env_path=Path(__file__).parent.parent.parent /".env"

load_dotenv(env_path)
api_key=os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Api Key Missing")
    exit()
    
else:
    print("Initalizing Model....")

model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',api_key=api_key,temperature=1.3)
embedding_model=GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')

text_splitter=SemanticChunker(
    embedding_model,breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

text="""
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.


Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.

"""

docs=text_splitter.create_documents([text])
print(len(docs))
print(docs)

prompt=PromptTemplate(
    template="Analyze the the given document-> \n {docs} and give answer to user as per these qsn->{qsn} and output should be standard chatbot type and should not contain ** and ###",
    input_variables=['docs','qsn']
)

chain = prompt | model | StrOutputParser()

while True:
    ask_model=input("\n\nEnter the qsn:")
    if ask_model.lower()=="end":
        break
    response=chain.invoke({'docs':text,'qsn':ask_model})
    print("\n\n\n\nResponse:",response)
    
