import os
import time
from langchain_groq import ChatGroq #type:ignore
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint  #type:ignore
from pathlib import Path
from dotenv import load_dotenv  #type:ignore
from langchain_core.prompts import PromptTemplate #type:ignore
from langchain_core.output_parsers import StrOutputParser #type:ignore
from langchain_core.runnables import RunnableParallel #type:ignore

env_path=Path(__file__).parent.parent/".env"
load_dotenv(env_path)
api_key=os.getenv('HUGGINGFACE_ACESS_TOKEN')
api_key2=os.getenv('GROQ_API_KEY')

if not api_key and  api_key2:
    print("Api_Key_Not_Found")
    exit()
    
else:
    print("Initalizing Model For You........")
  
#______________Model Setup____________________  
 
 #__________________ Initalizing Open Source model __________________________  
    
llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4.1-Flash",
    task="text_generation",
    huggingfacehub_api_token=api_key
)


model1=ChatHuggingFace(llm=llm)
time.sleep(2)


#___________________Initalizing Close Source Model________________________________

model2=ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=api_key2
)


#____________________________________________________________

prompt1=PromptTemplate(
    template="Generate a short and simple notes from the following text \n {text} ",
    input_variables=["text"]
)

prompt2=PromptTemplate(
    template="Generate 5 short question answer from following text \n {text}",
    input_variables=["text"]
)


prompt3=PromptTemplate(
    template="Merge the provided notes and quiz into a single documents \n  notes->{notes} and quiz->{quiz} ",
    input_variables=["notes","quiz"]
)

parser=StrOutputParser()


parallel_chain=RunnableParallel({
    'notes':prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain= prompt3 | model2 | parser


chain=parallel_chain | merge_chain

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""


result=chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()

# python3 parallel_chain.py 