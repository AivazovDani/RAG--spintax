import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
# read files from the specific directory

from langchain_text_splitters import CharacterTextSplitter
# chuck up the documents

from langchain_openai import OpenAIEmbeddings
# convert the chuncks into vector embedings

from langchain_chroma import Chroma
# import the Vector DB

from dotenv import load_dotenv
# loading all of the env variables from our file .env (like our OpenAi API key)


load_dotenv()

def load_documents(docs_path="docs"):
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist")
    
    loader = DirectoryLoader(
        path = docs_path,
        glob = "*.txt", # only look for txt files
        loader_cls = TextLoader # specific for txt files
    )

    documents = loader.load() # returns List[Document]

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files in {docs_path}")
    
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content lenght: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:100]}...")
        print(f" metadata: {doc.metadata}")

    return documents

def split_dicuments(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents) # split my docuemtns based on the chunck size

    if chunks:

        for i, chunk in enumerate(chunks[:5]):
            print(f"\n -----Chunk{i+1}------")
            print(f" Source: {chunk.metadata['source']}")
            print(f" Content lenght: {len(chunk.page_content)} characters")
            print("Content")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small") # it has 1,536 dimensions

    print("-----Creating Vector Store-------")

    vectorstore = Chroma.from_documents( # create a vector chrome database
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("-----Finished Embedding-----")

    return vectorstore


def main():
    #1. Loading the files
    documents = load_documents(docs_path="docs")
    #2. Chuncking files
    chunks = split_dicuments(documents)
    #3. Embedding the Storing in Vector DB
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()

