from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.text_splitter import CharacterTextSplitter
from redundant_filter_retriever import RedundantFilterRetriever
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

import langchain


langchain.debug = True
load_dotenv()

embeddings = OpenAIEmbeddings()
chat = OpenAI()

db = Chroma(embedding_function=embeddings, persist_directory="emb")

retriever = RedundantFilterRetriever(embeddings=embeddings, chroma=db)
# retriever = db.as_retriever(search_type="mmr")

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    # cheapest and used in most cases
    chain_type="stuff",
    # chain_type="map_reduce",
    # chain_type="map_rerank",
    # does not give good results as it uses the last output
    # chain_type="refine",
    # verbose is a bit bugged still
    # verbose=True,
)

result = chain.invoke("what is an interesting fact of the english language?")

print(result, "res")
