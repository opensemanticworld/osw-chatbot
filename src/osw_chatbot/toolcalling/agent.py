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
import pandas as pd
from pathlib import Path
DATA_PATH_DEFAULT = env_path = Path(__file__).parent.parent.parent.parent / "data"

import langchain_core

from llm import llm, embeddings

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

class PlotToolPanel():
    """a panel with callback functions to be called by a tool agent"""
    def __init__(self, data_path = None):
        if data_path is None:
            self.data_path = DATA_PATH_DEFAULT
        else:
            self.data_path = Path(data_path)

        self.fig, self.ax = plt.subplots()
        self.build_panel()

    def build_panel(self):
        self.frontend = ChatFrontendWidget()
        self.plot_panel = pn.pane.Matplotlib(self.fig)
        self._panel = pn.Row(self.plot_panel)
        return pn.Row(self.plot_panel, self.frontend)

    def load_data_from_csv(self) -> List[str]:
        """
        a function that loads data from a csv file
        returns the column names"""
        self.df = pd.read_csv(self.data_path / "test.csv", delimiter = "\t")
        return self.df.columns

    def plot_data(self, x_column_name: str, y_column_name: str) -> str:
        """
         a function that plots the data.
         returns "success" if the plot was successful, else the error message
        """
        print("bin am Anfang von plot_data")
        try:
            print(self.df)
            x = self.df[x_column_name]
            y = self.df[y_column_name]
            print(x,y)
            fig, ax = plt.subplots()
            ax.plot(x, y)
            self.plot_panel.object = fig
            print("habe geplottet")
            response = "success"
        except Exception as e:
            response = str(e)
        return response

    def __panel__(self):
        return self._panel

    def generate_langchain_tools(self)->List[langchain_core.tools.tool]:
        """
        returns a list of langchain tools that can be called by the agent
        """
        return [langchain_core.tools.tool(self.plot_data),
                langchain_core.tools.tool(self.load_data_from_csv)]

plot_tool_panel = PlotToolPanel()
plot_tools = plot_tool_panel.generate_langchain_tools()
print("plot_tools", plot_tools)

async def call_client_side_tool(toolcall):
    id = random.randint(0, 1000)
    toolcall["id"] = id
    plot_tool_panel.frontend.function_call = json.loads(json.dumps(toolcall))
    i = 0
    while i < 10: 
        print("waiting for response")
        await asyncio.sleep(1)
        print(plot_tool_panel.frontend.function_called)
        if (plot_tool_panel.frontend.function_called is not None and "id" in plot_tool_panel.frontend.function_called
                and
                plot_tool_panel.frontend.function_called["id"] == toolcall["id"]):
            return plot_tool_panel.frontend.function_called["result"]
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



tools = [multiply, redirect, find_page_from_topic, create_category_instance, *plot_tools]

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

