import os
from dotenv import load_dotenv 
from langchain_groq import ChatGroq 
from langchain_openai import ChatOpenAI 
from langchain_cohere import ChatCohere 
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser,JsonOutputParser
from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel,Field




# Load environment variables from .env file
load_dotenv()
api_key=os.getenv("OPENROUTER_API_KEY")
api_key2=os.getenv('GROQ_API_KEY')
api_key3=os.getenv("COHERE_API_KEY")
api_key4=os.getenv("OLLAMA_API_KEY")

if  not api_key and api_key2 and api_key3 and api_key4:
    print('Api Key Not Found')
    exit()

else:
    print("Initalizing Model .....")
    
groq_model=ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=api_key2
)

    
    
 
# 1. Initialize Ollama Cloud Model (using OpenAI-compatible interface)
ollama_model = ChatOpenAI(
    model="gemma4:31b",  # Ollama cloud model tag
    base_url="https://ollama.com/v1",  # Ollama cloud endpoint
    api_key=api_key4,
    temperature=0
)




# 2. Initialize Cohere Model
cohere_model = ChatCohere(
    model="command-r-plus-08-2024",  # Cohere model tag
    cohere_api_key=api_key3,
    temperature=0
)


# 3 Initialize Openrouter model

openrouter_model=ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key 
)

parser=StrOutputParser()

class Model_n_Company(BaseModel):
    model:str=Field(description='Model name  of the given model')
    company:str=Field(description='Parent company name of the given model')

m_c_parser=PydanticOutputParser(pydantic_object=Model_n_Company)
x=lambda res:{'model':res.model,'company':res.company}



prompt1=PromptTemplate(
    template="Whats your model name and actual  parent company name \n{format_instruction}",
    input_varaiables=[],
    partial_variables={'format_instruction':m_c_parser.get_format_instructions()}
)

prompt2=PromptTemplate(
    template="Whats is the  cutoff date the {model} that it was last trained from {company} \n display the input also",
    input_variables=['model','company']
)

prompt3=PromptTemplate(
    template=" In addition to this {text} Say Whats the capital of nepal \n display the input also",
    input_variables=['text']
)


prompt4=PromptTemplate(
    template="Summarize the given Info in 5 Lines. \n{text} \n display the input also",
    input_variables=['text']
)

parallel_chain=RunnableParallel({
    'chain_groq':prompt1 | groq_model | m_c_parser | x | prompt2 | groq_model | parser |prompt3 | groq_model | parser |prompt4 | groq_model | parser,
    'chain_cohere':prompt1 | cohere_model | m_c_parser | x | prompt2 | cohere_model | parser |prompt3 | cohere_model | parser |prompt4 | cohere_model | parser,
    'chain_openrouter':prompt1 | openrouter_model | m_c_parser | x | prompt2 | openrouter_model | parser |prompt3 | openrouter_model | parser |prompt4 | openrouter_model | parser
})


prompt5=PromptTemplate(
    template="""You are given a 3 result from three different model.Evalaute which model 
    answer is logical and real world acceptable.Rank them answer show their information(i.e  what input they  had given ).
    Important step: From the inputs of 3 model find out which model are they(i.e form their previous response or input that you were given)
    \n Models ->\n{chain_groq}\n{chain_cohere}\n{chain_openrouter}""",
    input_variables=['chain_groq','chain_cohere','chain_openrouter']
)

merge_chain= prompt5 | ollama_model | parser

chain=parallel_chain | merge_chain
result=chain.invoke({})

print(result)



chain.get_graph().print_ascii()


# python3 revisingchains_part1.py