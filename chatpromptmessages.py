from langchain_core.prompts import ChatPromptTemplate

chat_template=ChatPromptTemplate([
    ('system',"You are a helpful {domain} expert"),
    ('user',"Explain in simple terms, what is {user_query}")
    #SystemMessage(content="You are a helpful {domain} expert"),
    #HumanMessage(content="Explain in simple terms, what is {user_query}"),      
])

prompt=chat_template.invoke({
    'domain':'cricket',
    'user_query':'drs'
})

print(prompt)