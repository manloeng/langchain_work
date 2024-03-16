from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=0)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(text_splitter=text_splitter)

# Adds embeddings to db from docs
# everytime this is run, new embeddings are created and added to the db
# Which is why it looks like you have duplicated results
db = Chroma.from_documents(
    documents=docs, embedding=embeddings, persist_directory="emb"
)

results = db.similarity_search_with_score(
    "What is an interesting fact about a language?", k=3
)

for result in results:
    print("\n")
    print(result[1])
    print(result[0].page_content, "page_coontent")
