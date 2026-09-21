from langchain_community.document_loaders import PyPDFLoader
Filepath="/Users/pratiksha/Documents/PYTHON__XHIS_XHIS/Gen_Ai/nwee/helloo_there/samrat/Samrat/RAG/Documents_Loaders/dl-curriculum.pdf"

loader=PyPDFLoader(Filepath)

docs=loader.load()

print(docs[0].page_content)
print(docs[0].metadata)

print(len(docs))