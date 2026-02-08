from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
import re
import pprint

filepath="./example.pdf"

loader=PyPDFLoader(filepath)

docs=loader.load()

""" we use the RecursiveCharacterTextSplitter to maintain the context and paragraphs intact"""

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)

chunks=text_splitter.split_documents(docs)

for idx,chunk in enumerate(chunks):
  chunk.metadata["chunk_id"]=idx


embedings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

texts=[chunk.page_content for chunk in chunks]
metadata=[chunk.metadata for chunk in chunks]
ids=[str(chunk.metadata["chunk_id"]) for chunk in chunks]

vectors=embedings.embed_documents(texts)

client=chromadb.Client(Settings(persist_directory="./vectordb/chroma"))
collection=client.create_collection(name="my_collection")

vectors=embedings.embed_documents(texts)

collection.upsert(documents=texts,embeddings=vectors,metadatas=metadata,ids=ids)

def clean_text(text):
  text=re.sub(r'\s+',' ',text)
  text=re.sub(r'\.{3,}','',text)
  return text


samplequery="How does top management demonstrate leadership and commitment to the ISMS?"
sampletopk=5



def queryresponse(embedings,query,collection,topk):


  query_vector=embedings.embed_query(query)

  results=collection.query(
    query_embeddings=[query_vector],
    n_results=topk
  )

  retrived_docs=results["documents"][0]
  retrived_distances=results["distances"][0]
  ids=results["ids"][0]
  metadatas=results["metadatas"][0]
  distances=results["distances"][0]

  print(f"The Query : {query}")
  print(f"TOP K : {topk}")

  for index,(doc,dist,id,metadata,distance) in enumerate(zip(retrived_docs,retrived_distances,ids,metadatas,distances)):
    print("\n")
    print("-"*40)
    print(f"Result : {index+1}")
    print("\n")
    print(f"\n chunk ID :{id}")
    print("\n")
    print(f"Chunk Text : {doc}")
    print("\n")
    print(f"Page : {metadata['page']}")
    print("\n")
    print(f"Source : {metadata['source']}")
    print("\n")
    print(f"Score : {distance}")


print("Sample Outuput")
queryresponse(embedings,samplequery,collection,sampletopk)

print("\n")
print("The above output is the sample output Please enter your query below...")

print("\n\n")


query=input("Enter the Query :")
topk=int(input("Enter the TOP K value :"))
  
queryresponse(embedings,query,collection,topk)