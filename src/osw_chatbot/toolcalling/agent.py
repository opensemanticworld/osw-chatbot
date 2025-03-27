import asyncio
import json
import random
from base64 import b64decode
from pathlib import Path
from typing import Dict, List, Optional

import langchain_core
import langchain_core.tools
import matplotlib.pyplot as plt
import pandas as pd
import panel as pn
from langchain.agents import AgentExecutor, create_tool_calling_agent

# some Agent backend stuff
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from llm_sandbox import SandboxSession

from osw_chatbot.chat.chat_panel_component import ChatFrontendWidget
from osw_chatbot.llm import EMBEDDINGS, LLM

# Path setup
ROOT_DIR_PATH = Path(__file__).parents[3]
DOCKERFILE_SANDBOX_PATH = ROOT_DIR_PATH / "Dockerfile.sandbox"
DATA_PATH_DEFAULT = ROOT_DIR_PATH / "data"

vector_store = InMemoryVectorStore(EMBEDDINGS)


class OswFrontendPanel:
    """the class that will be handed over to OSW"""

    def __init__(self, child_panels=None):
        if child_panels is None:
            self.child_panels = []
        else:
            self.child_panels = child_panels
        self.build_panel()

    def build_panel(self):
        self.frontend = ChatFrontendWidget()
        self._panel = pn.Row(self.frontend, *self.child_panels)

    def __panel__(self):
        return self._panel

    async def call_client_side_tool(self, toolcall):
        id = random.randint(0, 1000)
        toolcall["id"] = id
        self.frontend.function_call = json.loads(json.dumps(toolcall))
        i = 0
        while i < 10:
            print("waiting for response")
            await asyncio.sleep(1)
            print(self.frontend.function_called)
            if (
                self.frontend.function_called is not None
                and "id" in self.frontend.function_called
                and self.frontend.frontend.function_called["id"]
                == toolcall["id"]
            ):
                return self.frontend.frontend.function_called["result"]
            i = i + 1
        print("timeout!")
        return None

    async def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""

        response = await self.call_client_side_tool(
            {"type": "function_call", "name": "multiply", "args": [a, b]}
        )
        return response

    async def redirect(self, page: str) -> str:
        """Redicts the user to the given page title or url. A page title must contain the namespace (e.g. 'Category:' or 'Item:'). Returns 'accepted' if the redirect was successful, else 'rejected'."""  # noqa

        response = await self.call_client_side_tool(
            {"type": "function_call", "name": "redirect", "args": [page]}
        )
        return response

    async def find_page_from_topic(self, topic) -> List[Dict[str, str]]:
        """Finds a page for a given topic for searching titles were
        the topic is contained in the label.
        Returns a list of results with title, description and type
        """
        response = await self.call_client_side_tool(
            {
                "type": "function_call",
                "name": "find_page_from_topic",
                "args": [topic],
            }
        )
        return response

    async def create_category_instance(self, category_page) -> str:
        """Opens an editor to create an instance for the given category page.
        Returns 'success' if the editor was opened, else 'failure'."""
        response = await self.call_client_side_tool(
            {
                "type": "function_call",
                "name": "create_category_instance",
                "args": [category_page],
            }
        )
        return response

    def generate_langchain_tools(self) -> List[langchain_core.tools.tool]:
        """
        returns a list of langchain tools that can be called by the agent
        """
        return [
            langchain_core.tools.tool(self.multiply),
            langchain_core.tools.tool(self.redirect),
            langchain_core.tools.tool(self.find_page_from_topic),
            langchain_core.tools.tool(self.create_category_instance),
        ]


class SandboxPlotPanel:
    """A tool to plot data by running code in sandbox environment"""

    def __init__(self, data_path=None):
        """Initializes the plot panel with a data path."""
        if data_path is None:
            self.data_path = DATA_PATH_DEFAULT
        else:
            self.data_path = Path(data_path)
        self.build_panel()

    def build_panel(self):
        """Build the panel."""
        self.image_panel = pn.pane.PNG()
        self._panel = self.image_panel

    def __panel__(self):
        return self._panel

    def generate_langchain_tools(self) -> List[langchain_core.tools.tool]:
        """
        returns a list of langchain tools that can be called by the agent
        """
        return [
            langchain_core.tools.tool(self.run_code),
        ]

    def run_code(
        self,
        lang: str,
        code: str,
        libraries: Optional[list] = None,
    ) -> str:
        """
        Run code in a sandboxed environment.
        :param lang: The language of the code.
        :param code: The code to run.
        :param libraries: The libraries to use, it is optional.
        :return: base64 encoded image of the plot.
        """

        with SandboxSession(
            # lang="python",
            lang=lang,
            libraries=libraries,
            dockerfile=DOCKERFILE_SANDBOX_PATH,
            keep_template=True,
        ) as session:
            # Run the code in the sandbox
            try:
                image_base64_str = session.run(code, libraries).text
                image_bytes = b64decode(image_base64_str)
                self.image_panel.object = image_bytes
                return "Image saved as base64 string"
            except Exception as e:
                return str(e)


class PlotToolPanel:
    """a panel with callback functions to be called by a tool agent"""

    def __init__(self, data_path=None):
        if data_path is None:
            self.data_path = DATA_PATH_DEFAULT
        else:
            self.data_path = Path(data_path)

        self.fig, self.ax = plt.subplots()
        self.build_panel()

    def build_panel(self):
        self.plot_panel = pn.pane.Matplotlib(self.fig)
        self._panel = self.plot_panel

    def load_data_from_csv(self) -> List[str]:
        """
        a function that loads data from a csv file
        returns the column names"""
        self.df = pd.read_csv(self.data_path / "test.csv", delimiter="\t")
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
            print(x, y)
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

    def generate_langchain_tools(self) -> List[langchain_core.tools.tool]:
        """
        returns a list of langchain tools that can be called by the agent
        """
        return [
            langchain_core.tools.tool(self.plot_data),
            langchain_core.tools.tool(self.load_data_from_csv),
        ]


class HistoryToolAgent:
    def __init__(self, tools, prompt_template=None):
        """
        a class including necessary functions to orchestrate a
        llm tool agent with history
        """

        self.tools = tools
        self.prompt_template = prompt_template
        if self.prompt_template is None:
            self.prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful assistant. If the user wants to create something first try to find the category page for the given topic/keyword the user metions. Then create the instance.",  # noqa
                    ),
                    ("placeholder", "{chat_history}"),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ]
            )

        # Construct the Tools agent
        self.langchain_agent = create_tool_calling_agent(
            llm=LLM, tools=tools, prompt=self.prompt_template
        )
        # Create an agent executor by passing in the agent and tools
        self.agent_executor = AgentExecutor(
            agent=self.langchain_agent, tools=tools, verbose=True
        )
        # agent_executor.invoke({"input": "what is LangChain?"})
        self.chat_history = []

    async def invoke(self, prompt):
        # return graph.invoke({"question": prompt})
        # return tools_llm.invoke(prompt)
        # return agent_executor.invoke({"input": prompt})
        res = await self.agent_executor.ainvoke(
            {"input": prompt, "chat_history": self.chat_history}
        )
        self.chat_history.extend(
            [
                HumanMessage(content=prompt),
                AIMessage(content=res["output"]),
            ]
        )
        return res
