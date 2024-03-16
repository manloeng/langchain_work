from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from pprint import pprint
import argparse

# Adds cli to pass values in
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Return a list of numbers")
parser.add_argument("--language", default="Python")
args = parser.parse_args()

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"],
)

code_chain = LLMChain(llm=OpenAI(), prompt=code_prompt)

res = code_chain.invoke({"language": args.language, "task": args.task})
pprint(res)
