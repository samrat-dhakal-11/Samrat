from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader,TextLoader,UnstructuredMarkdownLoader

loader=DirectoryLoader(
    path='/Users/pratiksha/Documents/PYTHON__XHIS_XHIS/Gen_Ai/nwee/helloo_there/samrat/Samrat/RAG/Documents_Loaders/books',
    glob='*.pdf',
    loader_cls=PyPDFLoader  #type:ignore
)
docs=loader.lazy_load()

print()


for document in docs:
    print()
    print(document.metadata)
    print()
