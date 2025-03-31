from typing import Dict, List
import asyncio
import json
import numpy as np

import random

import langchain_core.tools
import uuid

from osw.core import OSW, model
from osw_chatbot.chat.chat_panel_component import ChatFrontendWidget
from pydantic.v1 import BaseModel, Field
from typing import List, Dict, Tuple, Optional
from base64 import b64decode
import panel as pn
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from PIL import Image

import io
from datetime import datetime

from osw.controller.file.local import LocalFileController
from osw.controller.file.wiki import WikiFileController
from osw.utils.wiki import get_full_title

from osw.express import osw_upload_file, OswExpress

import langchain_core

from panel.io.mime_render import exec_with_return
from llm_sandbox import SandboxSession
from llm import llm, embeddings

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

DATA_PATH_DEFAULT = env_path = Path(__file__).parent.parent.parent.parent / "data"

class OswFrontendPanel():
    """the class that will be handed over to OSW"""
    def __init__(self, child_panels = None):
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

    async def call_client_side_tool(self,toolcall):
        id = random.randint(0, 1000)
        toolcall["id"] = id
        self.frontend.function_call = json.loads(json.dumps(toolcall))
        i = 0
        while i < 10:
            print("waiting for response")
            await asyncio.sleep(1)
            print(self.frontend.function_called)
            if (
                    self.frontend.function_called is not None and "id" in self.frontend.function_called
                    and
                    self.frontend.function_called["id"] == toolcall["id"]):
                return self.frontend.function_called["result"]
            i = i + 1
        print("timeout!")
        return None

    async def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""

        response = await self.call_client_side_tool({"type": "function_call", "name": "multiply", "args": [a, b]})
        return response

    async def redirect(self, page: str) -> str:
        """Redicts the user to the given page title or url. A page title must contain the namespace (e.g. 'Category:' or 'Item:'). Returns 'accepted' if the redirect was successful, else 'rejected'."""

        response = await self.call_client_side_tool({"type": "function_call", "name": "redirect", "args": [page]})
        return response

    async def find_page_from_topic(self, topic) -> List[Dict[str, str]]:
        """Finds a page for a given topic for searching titles were the topic is contained in the label.
        Returns a list of results with title, description and type
        """
        response = await self.call_client_side_tool(
            {"type": "function_call", "name": "find_page_from_topic", "args": [topic]})
        return response

    async def create_category_instance(self, category_page) -> str:
        """Opens an editor to create an instance for the given category page. Returns 'success' if the editor was opened, else 'failure'."""
        response = await self.call_client_side_tool(
            {"type": "function_call", "name": "create_category_instance", "args": [category_page]})
        return response

    async def resize_chatwindow(self):
        """resizes the chatbot window"""
        response = await self.call_client_side_tool(
            {"type": "function_call", "name": "resize_chatwindow", "args": []})
        return response

    def generate_langchain_tools(self)->List[langchain_core.tools.tool]:
        """
        returns a list of langchain tools that can be called by the agent
        """
        return [#langchain_core.tools.tool(self.multiply),
                langchain_core.tools.tool(self.redirect),
                #langchain_core.tools.tool(self.find_page_from_topic),
                langchain_core.tools.tool(self.create_category_instance),
                langchain_core.tools.tool(self.resize_chatwindow),
                ]

class PlotConfig(BaseModel):
    x_column_name: str
    y_column_name: str
    x_label: str = None
    y_label: str = None





class LoadDataFromCsvInput(BaseModel):
    file_path: str = Field(..., description="The path to the file to read to a pandas dataframe.")
    delimiter: str = Field(default="\t", description="The delimiter of the csv file.")
    skip_rows: int = Field(default=0, description="The number of lines to skip at the beginning of the file.")



class AttachCurrentPlotToOswPageInput(BaseModel):
    osw_id: str = Field(..., description="Fullpagetitle of the page to attach the plot to. It has to be formatted like "
                                         "<NAMESPACE>:<OSW_ID> for example 'Item:OSW0b80ad413e954c87ac48bcc6ed784276' or 'Category:OSW0b80ad413e954c87ac48bcc6ed784276'.")
    format: str = Field(default="png", description="The format to save the plot in.")

def read_local_csv_to_df(file_path: str):
    """
    reads a local file to a pandas dataframe
    """
    df = pd.read_csv(file_path)

class PlotByCodeInput(BaseModel):
    lang: str = Field(default="python", description="The language code of the page.")
    code: str = Field(..., description="The code to run. The code must save a figure as .png into a io.BytesIO object "
                                       "and convert it to a base64 encoded string. Print this string to console ("
                                       "but nothing else).")
    file_path: Optional[str] = Field(default=None, description="The path to a file that can be used within the "
                                                              "script. The path from within the script must be "
                                                              "'/sandbox/<FILENAME>' where <FILENAME> can be "
                                                               "extracted from from the <file_path>")
    libraries: Optional[List[str]] = Field(default=["numpy", "pandas", "matplotlib", "scipy"], description=("The "
                                                                                                           "libraries to use. Only change if more than default is needed."))

class RunCodeInput(BaseModel):
    lang: str = Field(default="python", description="The language code of the page.")
    code: str = Field(..., description="The code to run. Whatever is printed will be returned.")
    file_path: Optional[str] = Field(default=None, description="The path to a file that can be used within the "
                                                              "script. The path from within the script must be "
                                                              "'/sandbox/<FILENAME>' where <FILENAME> can be "
                                                               "extracted from from the <file_path>")
    libraries: Optional[List[str]] = Field(default=["numpy", "pandas", "matplotlib", "scipy"], description="The "
                                                                                                           "libraries to use. Only change if more than default is needed.")
    file_type: Optional[str] = Field(default=None, description="The type of the file to use from filepath.")

class PlotToolPanel():
    """a panel with callback functions to be called by a tool agent"""
    def __init__(self, data_path = None):
        if data_path is None:
            self.data_path = DATA_PATH_DEFAULT
        else:
            self.data_path = Path(data_path)
        self.df=None
        self.fig, self.ax = plt.subplots()
        self.build_panel()

    def build_panel(self):

      #  self.matplotlib_panel = pn.pane.Matplotlib(self.fig, width = 600)
        self.image_panel = pn.pane.Image(self.fig, width = 600)
        self.plot_panel = pn.Row(self.image_panel)
        self._panel = self.plot_panel

    def load_data_from_csv(self, inp:LoadDataFromCsvInput) -> List[str]:
        """
        a function that loads data from a csv file
        returns the column names"""
        self.df = pd.read_csv(inp.file_path, delimiter = inp.delimiter, skiprows=inp.skip_rows)
        return self.df.columns


    # def plot_data(self, plot_config: PlotConfig) -> str:
    #     """
    #      returns "success" if the plot was successful, else the error message
    #     """
    #
    #     try:
    #         print(self.df)
    #         x = self.df[plot_config.x_column_name]
    #         y = self.df[plot_config.y_column_name]
    #
    #         fig, ax = plt.subplots()
    #         ax.plot(x, y)
    #
    #         if plot_config.x_label is not None:
    #             ax.set_xlabel(plot_config.x_label)
    #         if plot_config.y_label is not None:
    #             ax.set_ylabel(plot_config.y_label)
    #
    #         self.matplotlib_panel.object = fig
    #         self.plot_panel.clear()
    #         self.plot_panel.append(self.matplotlib_panel)
    #         response = "success"
    #     except Exception as e:
    #         response = str(e)
    #     return response

    def plot_by_code(
        self, inp: PlotByCodeInput

    ) -> str:
        """
        Run code in a sandboxed environment. If files are needed they can be copied to the sandbox with the file_path parameter.
        """

        with SandboxSession(
            # lang="python",
            lang=inp.lang,
            libraries=inp.libraries,
            image="python:3.12-slim",
            # dockerfile=DOCKERFILE_SANDBOX_PATH,
            #verbose=True,
            keep_template=True,
        ) as session:
            # Run the code in the sandbox
            try:
                if True:#inp.file_path is not None:
                    filename = Path(inp.file_path).name
                    dest_filepath = "/sandbox/" + filename
                    session.copy_to_runtime(src = inp.file_path,
                                            dest = dest_filepath,)
                image_base64_str = session.run(inp.code, inp.libraries).text
                print("len(image_base64_str)", len(image_base64_str))
                code_path = DATA_PATH_DEFAULT / "plot_codes" /"code.py"
                session.copy_from_runtime(src="/tmp/code.py", dest = str(code_path))

                ### check if the base64 string encodes an image:
                image_bytes = b64decode(image_base64_str)
                Image.open(io.BytesIO(image_bytes))# open the image to check if the file is correct.

                self.image_panel.object = image_bytes
                self.plot_panel.clear()
                self.plot_panel.append(self.image_panel)
                return "Image successfully plotted"
            except Exception as e:
                return("the script did not return a base64 encoded image, instead it returned: " + str(e),
                       "\ the string above is from the except case.")

    def run_code(
        self, inp: RunCodeInput

    ) -> str:
        """
        Run code in a sandboxed environment. Typacilly used to create text from code. Return whatever has been
        printed to the console within the python code.
        """

        with SandboxSession(
            # lang="python",
            lang=inp.lang,
            libraries=inp.libraries,
            image="python:3.12-slim",
            # dockerfile=DOCKERFILE_SANDBOX_PATH,
            #verbose=True,
            keep_template=True,
        ) as session:
            # Run the code in the sandbox
            try:
                if True:#inp.file_path is not None:
                    filename = Path(inp.file_path).name
                    dest_filepath = "/sandbox/" + filename
                    session.copy_to_runtime(src = inp.file_path,
                                            dest = dest_filepath,)
                return_str = session.run(inp.code, inp.libraries).text
                print("len(image_base64_str)", len(return_str))
                code_path = DATA_PATH_DEFAULT / "run_codes" /"code.py"
                session.copy_from_runtime(src="/tmp/code.py", dest = str(code_path))

                return return_str
            except Exception as e:
                return(str(e))


    def attach_current_plot_to_osw_page(self, inp: AttachCurrentPlotToOswPageInput):
        """uploads the current plot to the osw page with the given id"""
        ret_msg = ""
        try:
            ## get the page object where the plot should be attached

            osw_obj = OswExpress(domain="mat-o-lab.open-semantic-lab.org")
            title = inp.osw_id
            entity = osw_obj.load_entity(title)
            if entity is None:
                return "error loading entity with title: " + title + " was it formatted correctly?"
            ## save plot to bytesio object:
            bytesio = io.BytesIO()
            self.plot_panel.object.savefig(bytesio, format="png")
            plot_uuid = uuid.uuid4()
            wf = WikiFileController(uuid=str(plot_uuid), osw=osw_obj,
                                    title="OSW" + str(plot_uuid).replace("-", "") + ".png",
                                    label=[model.Label(text="Plot from Chatbot " + str(datetime.now().strftime(
                                        '%Y-%m-%d_%H-%M')))])
            bytesio.name = wf.title
            try:
                wf.put(bytesio, overwrite=True)
                ret_msg += "plot successfully uploaded to osw page with uuid: " + str(wf.uuid)
            except Exception as e:
                ret_msg += "error uploading plot to osw page: " + str(e)
            ## link it to the page:
            entity.attachments.append(get_full_title(wf))
            ## re-upload entity:
            osw_obj.store_entity(OSW.StoreEntityParam(entities=[entity], overwrite=True))
            ret_msg += "attachment added successfully, "
            return  ret_msg
        except Exception as e:
            return str(e)
    def __panel__(self):
        return self._panel

    def generate_langchain_tools(self)->List[langchain_core.tools.tool]:
        """
        returns a list of langchain tools that can be called by the agent
        """


        return [#langchain_core.tools.tool(self.plot_data),
                langchain_core.tools.tool(self.load_data_from_csv),
                langchain_core.tools.tool(self.attach_current_plot_to_osw_page),
                langchain_core.tools.tool(self.plot_by_code),
                langchain_core.tools.tool(self.run_code),
                ]


### some Agent backend stuff
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

class HistoryToolAgent():
    def __init__(self, tools, prompt_template = None):

        """
        a class including necessary functions to orchestrate a llm tool agent with history
        """

        self.tools = tools
        self.prompt_template = prompt_template
        if self.prompt_template is None:
            self.prompt_template = ChatPromptTemplate.from_messages(
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
        self.langchain_agent = create_tool_calling_agent(llm, tools, self.prompt_template)
        # Create an agent executor by passing in the agent and tools
        self.agent_executor = AgentExecutor(agent=self.langchain_agent,
                                            tools=tools,
                                            verbose=True,
                                            return_intermediate_steps = True)
        #agent_executor.invoke({"input": "what is LangChain?"})
        self.chat_history = []


    async def invoke(self,prompt):
        #return graph.invoke({"question": prompt})
        #return tools_llm.invoke(prompt)
        #return agent_executor.invoke({"input": prompt})
        res = await self.agent_executor.ainvoke({"input": prompt, "chat_history": self.chat_history})
        print("Whole result of invoke", res)
        print ("Useful parts of result: ", res["intermediate_steps"])
        self.chat_history.extend([
            HumanMessage(content=prompt),
            str(res["intermediate_steps"]),
            AIMessage(content=res["output"]),
        ])
        return res


