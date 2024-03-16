from langchain.chains import LLMChain
from langchain.memory import (
    ConversationBufferMemory,
    FileChatMessageHistory,
    ConversationSummaryMemory,
)
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(verbose=True)

key = "messages"
# memory = ConversationBufferMemory(
#     chat_memory=FileChatMessageHistory("message.json"),
#     memory_key=key,
#     return_messages=True,
# )
memory = ConversationSummaryMemory(
    chat_memory=FileChatMessageHistory("message.json"),
    memory_key=key,
    return_messages=True,
    llm=chat,
)

prompt = ChatPromptTemplate(
    input_variables=["content", key],
    messages=[
        MessagesPlaceholder(variable_name=key),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

chain = LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True).invoke

while True:
    content = input(">> ")
    result = chain({"content": content})
    print(result["text"])
