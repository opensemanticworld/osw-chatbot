from pathlib import Path
from typing import Dict
import asyncio
import getpass
import json

import random
from time import sleep

from attr import has
import langchain_core.tools

import langchain_core

from osw_chatbot.llm import llm, embeddings

from langchain_core.vectorstores import InMemoryVectorStore


import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

vector_store = InMemoryVectorStore(embeddings)


def index(input_file: Path):
    docs = TextLoader(file_path=input_file, encoding="utf-8").load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    # Index chunks
    _ = vector_store.add_documents(documents=all_splits)


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    answer = response  # llama
    if hasattr(answer, "content"):
        answer = answer.content  # openai
    return {"answer": answer}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
