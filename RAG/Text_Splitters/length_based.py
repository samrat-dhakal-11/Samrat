from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

loader=PyPDFLoader('dl-curriculum.pdf')
docs=loader.load()


splitter=CharacterTextSplitter(
    chunk_size=00,
    chunk_overlap=0,
    separator=''
)

result=splitter.split_documents(docs)

print(len(result))
print("1st chunk is:\n",result[0].page_content)

