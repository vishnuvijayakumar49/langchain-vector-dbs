import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
  print("Ingesting data...")
  loader = TextLoader("/Users/vishnuvijayakumar/Projects/AI/vector-dbs-intro/mediumblog1.txt")
  documents = loader.load()

  print("splitting the document into chunks...")
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  texts = text_splitter.split_documents(documents)
  print(f"Created {len(texts)} chunks")
  
  embeddings = OpenAIEmbeddings(
    check_embedding_ctx_length=False,
    base_url="http://localhost:1234/v1",
    api_key="key",
    model="text-embedding-nomic-embed-text-v1.5"
  )
  # vector_store = Chroma(embedding_function=embed_model)
  vector_store = PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"])
  print("Finished creating vector store")
