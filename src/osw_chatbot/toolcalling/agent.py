from typing import Dict, List
import asyncio
import json
import numpy as np

import random

import langchain_core.tools
from osw_chatbot.chat.chat_panel_component import ChatFrontendWidget
from pydantic.v1 import BaseModel
from typing import Tuple
import panel as pn
import matplotlib.pyplot as plt

import langchain_core

from llm import llm, embeddings

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

frontend =  ChatFrontendWidget()


fig, ax = plt.subplots()
plot_panel = pn.pane.Matplotlib(fig)


async def call_client_side_tool(toolcall):
    id = random.randint(0, 1000)
    toolcall["id"] = id
    frontend.function_call = json.loads(json.dumps(toolcall))
    i = 0
    while i < 10: 
        print("waiting for response")
        await asyncio.sleep(1)
        print(frontend.function_called)
        if frontend.function_called is not None and "id" in frontend.function_called and frontend.function_called["id"] == toolcall["id"]:
            return frontend.function_called["result"]
        i = i + 1
    print("timeout!")
    return None

@langchain_core.tools.tool
async def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    
    response = await call_client_side_tool({"type": "function_call", "name": "multiply", "args": [a, b]})
    return response

@langchain_core.tools.tool
async def redirect(page: str) -> str:
    """Redicts the user to the given page title or url. A page title must contain the namespace (e.g. 'Category:' or 'Item:'). Returns 'accepted' if the redirect was successful, else 'rejected'."""
    
    response = await call_client_side_tool({"type": "function_call", "name": "redirect", "args": [page]})
    return response

@langchain_core.tools.tool
async def find_page_from_topic(topic) -> List[Dict[str, str]]:
    """Finds a page for a given topic for searching titles were the topic is contained in the label.
    Returns a list of results with title, description and type
    """
    response = await call_client_side_tool({"type": "function_call", "name": "find_page_from_topic", "args": [topic]})
    return response

@langchain_core.tools.tool
async def create_category_instance(category_page) -> str:
    """Opens an editor to create an instance for the given category page. Returns 'success' if the editor was opened, else 'failure'."""
    response = await call_client_side_tool({"type": "function_call", "name": "create_category_instance", "args": [category_page]})
    return response



@langchain_core.tools.tool
async def plot_test() -> Tuple[np.ndarray, np.ndarray]:
    """a plot funciton for testing
    returns a tuple of two lists x and y
    """
    x = np.linspace(0,10,100)
    y = np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    plot_panel.object = fig
    print("habe geplottet")
    response = (x,y)
    return response

tools = [multiply, redirect, find_page_from_topic, create_category_instance, plot_test]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. If the user wants to create something first try to find the category page for the given topic/keyword the user metions. Then create the instance.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


# Construct the Tools agent
agent = create_tool_calling_agent(llm, tools, prompt)
# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#agent_executor.invoke({"input": "what is LangChain?"})

from langchain_core.messages import AIMessage, HumanMessage
chat_history = []

async def invoke(prompt):
    #return graph.invoke({"question": prompt})
    #return tools_llm.invoke(prompt)
    #return agent_executor.invoke({"input": prompt})
    res = await agent_executor.ainvoke({"input": prompt, "chat_history": chat_history})
    chat_history.extend([
        HumanMessage(content=prompt),
        AIMessage(content=res["output"]),
    ])
    return res

