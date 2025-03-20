import os

from langchain_community.cache import SQLiteCache
from langchain_ollama import OllamaEmbeddings
from langchain_openai import AzureChatOpenAI

LLM = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    # azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    model="gpt-4o-2024-08-06",
    cache=SQLiteCache("openai_cache.db"),
)

# from tools import tools
# tools_llm = llm.bind_tools(tools)

# from langchain_openai import AzureOpenAIEmbeddings

# embeddings = AzureOpenAIEmbeddings(
#     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
#     #azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
#     api_version=os.environ["AZURE_OPENAI_API_VERSION"],
#     model="text-embedding-ada-002-2",
# )


# llm = OllamaLLM(
#     model="deepseek-r1:7b",
# )

EMBEDDINGS = OllamaEmbeddings(
    model="nomic-embed-text",
)
