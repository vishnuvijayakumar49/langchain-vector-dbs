import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
  print("Starting retriving...")

  embed_model = OpenAIEmbeddings(
    check_embedding_ctx_length=False,
    base_url="http://localhost:1234/v1",
    api_key="key",
    model="text-embedding-nomic-embed-text-v1.5"
  )

  llm = ChatGoogleGenerativeAI(model= 'gemini-1.5-flash')
  query = "What is pinecone in machine learning?"
  vector_store = PineconeVectorStore(embedding=embed_model, index_name=os.environ["INDEX_NAME"])
  
  retrival_qa_chat_prompt = """Answer the following question based on the provided context:

  Context: {context}

  Question: {input}

  Answer: """

  custom_prompt = """Use the following pieces of retrieved context to answer the question. 
  If you don't have enough information, just say you don't know, don't make up an answer.
  Use three sentences maximum and keep it concise as possible.
  always say "thank you asking!" at the end of your answer.

  {context}

  Question: {question}

  Helpful Answer: """
  
  custom_rag_prompt_template = PromptTemplate.from_template(custom_prompt)
  retrival_chain = (
    {"context": vector_store.as_retriever() | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt_template
    | llm
  )

  result = retrival_chain.invoke(query)
  print(result)
