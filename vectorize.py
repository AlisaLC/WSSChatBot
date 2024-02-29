from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

import dotenv
import pickle

if __name__ == "__main__":
    dotenv.load_dotenv()
    loader = UnstructuredFileLoader("datasets/Q&A.txt")
    splitter = RecursiveCharacterTextSplitter(
        separators='\n\n',
        chunk_size=20,
        chunk_overlap=0,
        length_function=len,
    )
    documents = loader.load_and_split(splitter)
    for doc in documents:
        print(doc.page_content.strip())
        print('-' * 80)
    vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
    with open("datasets/vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)