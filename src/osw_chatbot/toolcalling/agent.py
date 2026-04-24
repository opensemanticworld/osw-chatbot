from typing import Dict, List
import asyncio
import json
import numpy as np

import random
import os
import langchain_core.tools
import uuid
import warnings

from numba.core.ir import Raise
from pydantic import PydanticDeprecatedSince211
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince211)
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

from osw.express import osw_upload_file, OswExpress, default_paths

import langchain_core

from panel.io.mime_render import exec_with_return
from llm_sandbox import SandboxSession
from llm import llm, embeddings

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

DATA_PATH_DEFAULT = env_path = (
    Path(__file__).parent.parent.parent.parent / "data"
)

def _resolve_file_path(file_path: str) -> str:
    """Resolve a file path that may be just a filename to its full host path."""
    p = Path(file_path)
    if p.is_absolute() and p.exists():
        return str(p)
    candidate = default_paths.download_dir / p.name
    if candidate.exists():
        return str(candidate)
    candidate = DATA_PATH_DEFAULT / p.name
    if candidate.exists():
        return str(candidate)
    return file_path


_MICRESS_BIN_EXTS = {".conc1", ".conc2", ".concc", ".phas", ".temp", ".vel"}
_MICRESS_GEO_EXTS = {".geof", ".geof1"}


def _copy_files_to_sandbox(session, file_paths: List[str]) -> List[str]:
    """Copy files into the sandbox, renaming geometry files to match binary
    files so that micpy auto-detects geometry."""
    resolved_paths = [_resolve_file_path(fp) for fp in file_paths]

    bin_stem = None
    geo_src = None
    non_geo = []

    for rp in resolved_paths:
        suffix = Path(rp).suffix.lower()
        if suffix in _MICRESS_GEO_EXTS:
            geo_src = rp
        else:
            if suffix in _MICRESS_BIN_EXTS:
                bin_stem = Path(rp).stem
            non_geo.append(rp)

    filenames = []
    for rp in non_geo:
        name = Path(rp).name
        session.copy_to_runtime(src=rp, dest="/sandbox/" + name)
        filenames.append(name)

    if geo_src is not None:
        if bin_stem is not None:
            geo_name = bin_stem + ".geoF"
        else:
            geo_name = Path(geo_src).name
        session.copy_to_runtime(src=geo_src, dest="/sandbox/" + geo_name)
        filenames.append(geo_name)

    return filenames

class WebPage(BaseModel):
    url: str
    """The full url of the web page."""
    title: str
    """The title of the web page."""
    content: str
    """The html content of the page."""

class OswFrontendPanel:
    """the class that will be handed over to OSW"""

    def __init__(self, child_panels=None, resize_callback=None):
        """
        child_panels:
        """
        if child_panels is None:
            self.child_panels = []
        else:
            self.child_panels = child_panels
        self.resize_callback = resize_callback
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
                and self.frontend.function_called["id"] == toolcall["id"]
            ):
                return self.frontend.function_called["result"]
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
        """Redicts the user to the given page title or url.
        A page title must contain the namespace (e.g. 'Category:' or 'Item:').
        Returns 'accepted' if the redirect was successful, else 'rejected'."""

        response = await self.call_client_side_tool(
            {"type": "function_call", "name": "redirect", "args": [page]}
        )
        return response

    async def find_page_from_topic(self, topic) -> List[Dict[str, str]]:
        """Finds a page for a given topic for searching titles were the topic is contained in the label.
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

    async def resize_chatwindow(self):
        """resizes the chatbot window"""
        response = await self.call_client_side_tool(
            {"type": "function_call", "name": "resize_chatwindow", "args": []}
        )
        if self.resize_callback is not None:
            self.resize_callback(response)
        return response

    async def where_am_i(self) -> WebPage:
        """Returns the current window.location url, the document.title and body html of the users browser client."""

        response = await self.call_client_side_tool({"type": "function_call", "name": "where_am_i", "args": []})
        response = WebPage(**response)
        return response

    def generate_langchain_tools(self) -> List[langchain_core.tools.tool]:
        """
        returns a list of langchain tools that can be called by the agent
        """
        return [  # langchain_core.tools.tool(self.multiply),
            langchain_core.tools.tool(self.redirect),
            langchain_core.tools.tool(self.where_am_i), # currently working in dev wiki
            # langchain_core.tools.tool(self.find_page_from_topic),
            langchain_core.tools.tool(self.create_category_instance),
            langchain_core.tools.tool(self.resize_chatwindow),
        ]


class PlotConfig(BaseModel):
    x_column_name: str
    y_column_name: str
    x_label: str = None
    y_label: str = None


class LoadDataFromCsvInput(BaseModel):
    file_path: str = Field(
        ..., description="The path to the file to read to a pandas dataframe."
    )
    delimiter: str = Field(
        default="\t", description="The delimiter of the csv file."
    )
    skip_rows: int = Field(
        default=0,
        description="The number of lines to skip at the beginning of the file.",
    )


class AttachCurrentPlotToOswPageInput(BaseModel):
    osw_id: str = Field(
        ...,
        description="Fullpagetitle of the page to attach the plot to. It has to be formatted like "
        "<NAMESPACE>:<OSW_ID> for example 'Item:OSW0b80ad413e954c87ac48bcc6ed784276' or 'Category:OSW0b80ad413e954c87ac48bcc6ed784276'.",
    )
    format: str = Field(
        default="png", description="The format to save the plot in."
    )


class DocumentCurrentEvaluationInput(BaseModel):
    uuid: Optional[str] = Field(
        ...,
        description="The uuid of the evaluation process. A new uuid is generated if None",
    )
    output_osw_id: Optional[str] = Field(
        ...,
        description="The OSW ID of the output of the evaluation process, e.g. the OSW-Object plot.",
    )


def read_local_csv_to_df(file_path: str):
    """
    reads a local file to a pandas dataframe
    """
    df = pd.read_csv(file_path)


class PlotByCodeInput(BaseModel):
    lang: str = Field(
        default="python", description="The language code of the page."
    )
    code: str = Field(
        ...,
        description='The code to run. The code must save a figure as .png to a file at "/sandbox/output.png". The code '
                    'must be utf-8 encoded. The code must be able to run in a sandboxed environment.',
    )
    file_paths: Optional[List[str]] = Field(
        default=None,
        description="List of file paths to copy into the sandbox before running "
        "the code. Each file will be available at '/sandbox/<FILENAME>' "
        "where <FILENAME> is the basename of the path. Accepts full host "
        "paths or just filenames (resolved from the downloads directory).",
    )
    libraries: Optional[List[str]] = Field(
        default=["numpy", "pandas", "matplotlib", "scipy"],
        description=(
            "List of pip packages to install in the sandbox before running the code. "
            "The defaults are numpy, pandas, matplotlib, scipy. Add any additional "
            "packages the code needs (e.g. 'micress-micpy', 'seaborn', 'scikit-learn'). "
            "Note: the pip package name may differ from the Python import name."
        ),
    )


class RunCodeInput(BaseModel):
    lang: str = Field(
        default="python", description="The language code of the page."
    )
    code: str = Field(
        ...,
        description="The code to run. Whatever is printed will be returned.",
    )
    file_paths: Optional[List[str]] = Field(
        default=None,
        description="List of file paths to copy into the sandbox before running "
        "the code. Each file will be available at '/sandbox/<FILENAME>' "
        "where <FILENAME> is the basename of the path. Accepts full host "
        "paths or just filenames (resolved from the downloads directory).",
    )
    libraries: Optional[List[str]] = Field(
        default=["numpy", "pandas", "matplotlib", "scipy"],
        description=(
            "List of pip packages to install in the sandbox before running the code. "
            "The defaults are numpy, pandas, matplotlib, scipy. Add any additional "
            "packages the code needs (e.g. 'micress-micpy', 'seaborn', 'scikit-learn'). "
            "Note: the pip package name may differ from the Python import name."
        ),
    )


class PlotToolPanel:
    """a panel with callback functions to be called by a tool agent"""

    def __init__(self, data_path=None):
        if data_path is None:
            self.data_path = DATA_PATH_DEFAULT
        else:
            self.data_path = Path(data_path)
        self.df = None
        self.fig, self.ax = plt.subplots()
        self.build_panel()
        self.current_input_osw_id = None

    def build_panel(self):
        #  self.matplotlib_panel = pn.pane.Matplotlib(self.fig, width = 600)
        self.image_panel = pn.pane.Image(self.fig, width=600)
        self.plot_panel = pn.Row(self.image_panel)
        self._panel = self.plot_panel

    def load_data_from_csv(self, inp: LoadDataFromCsvInput) -> List[str]:
        """
        a function that loads data from a csv file
        returns the column names"""
        self.df = pd.read_csv(
            inp.file_path, delimiter=inp.delimiter, skiprows=inp.skip_rows
        )
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

    def plot_by_code(self, inp: PlotByCodeInput) -> str:
        """
        Run code in a sandboxed environment. If files are needed they can be copied
        to the sandbox with the file_path parameter.
        The plot must be saved as a .png file to "/sandbox/output.png"
        """
        return_str = None
        with SandboxSession(
            lang="python",
            image="python:3.12-slim",
            keep_template=True,
        ) as session:
            try:
                filenames = []
                if inp.file_paths is not None:
                    filenames = _copy_files_to_sandbox(session, inp.file_paths)
                return_str = session.run(inp.code, inp.libraries).stdout
                code_path = DATA_PATH_DEFAULT / "plot_codes" / "code.py"
                code_path.parent.mkdir(parents=True, exist_ok=True)
                code_path.write_text(inp.code, encoding="utf-8")
                self.output_file_path = DATA_PATH_DEFAULT / "outputs" / "output.png"
                self.output_file_path.parent.mkdir(parents=True, exist_ok=True)
                session.copy_from_runtime(
                    src="/sandbox/output.png", dest=str(self.output_file_path)
                )
                ### check if the output file encodes an image:
                # open the image to check if the file is correct.
                Image.open(
                    self.output_file_path
                )
                self.image_panel.object = None  ## to trigger re-plotting
                self.image_panel.object = str(self.output_file_path)
                self.plot_panel.clear()
                self.plot_panel.append(self.image_panel)
                if filenames:
                    self.current_input_osw_id = (
                        "File:" + filenames[0]
                    )
                else:
                    self.current_input_osw_id = None
                # bytesio
                # objects are used for intermediary storage

                ## copy code to current Object
                self.current_python_code = inp.code
                return "Image successfully plotted, returned from sandbox: " + return_str
            except Exception as e:
                if return_str is not None:
                    return (
                        "Exception during plotting: "
                        + str(e)
                        + "\n"
                        + "Returned from sandbox: "
                        + return_str
                    )
                return (
                    "Exception during plotting: "
                    + str(e),
                )

    def run_code(self, inp: RunCodeInput) -> str:
        """
        Run code in a sandboxed environment. Typacilly used to create text from code. Return whatever has been
        printed to the console within the python code.
        """

        with SandboxSession(
            lang=inp.lang,
            image="python:3.12-slim",
            keep_template=True,
        ) as session:
            try:
                if inp.file_paths is not None:
                    _copy_files_to_sandbox(session, inp.file_paths)
                return_str = session.run(inp.code, inp.libraries).stdout
                code_path = DATA_PATH_DEFAULT / "run_codes" / "code.py"
                code_path.parent.mkdir(parents=True, exist_ok=True)
                code_path.write_text(inp.code, encoding="utf-8")

                return return_str

            except Exception as e:
                return str(e)

    def document_current_evaluation(self, inp: DocumentCurrentEvaluationInput):
        """documents the evaluation of something with a python snippet"""

        ret_msg = ""
        try:
            osw_obj = OswExpress(domain=os.environ.get("OSW_DOMAIN"))

            ## upload the image and attach it as output
            # upload the image:

            ## save plot to bytesio object:
            if self.plot_panel[0] == self.image_panel:
                if isinstance(self.image_panel.object, str):
                    with open(self.image_panel.object, 'rb') as file:
                        bytesio = io.BytesIO(file.read())
                elif isinstance(self.image_panel.object, bytes):
                    bytesio = io.BytesIO(self.image_panel.object)
                else:
                    raise ValueError("No image to attach")

            elif self.plot_panel[0] == self.matplotlib_panel:
                bytesio = io.BytesIO()
                self.matplotlib_panel.object.savefig(bytesio, format="png")
            else:
                return "no plot to attach"
            plot_uuid = uuid.uuid4()
            wf = WikiFileController(
                uuid=str(plot_uuid),
                osw=osw_obj,
                title="OSW" + str(plot_uuid).replace("-", "") + ".png",
                label=[
                    model.Label(
                        text="Plot from Chatbot "
                        + str(datetime.now().strftime("%Y-%m-%d_%H-%M"))
                    )
                ],
            )
            bytesio.name = wf.title
            try:
                wf.put(bytesio,
                       overwrite=True)
            except Exception as e:
                ret_msg += "error uploading plot to osw page: " + str(e)

            ## link it to the documentation object:
            if self.current_input_osw_id is not None:
                input_list = [self.current_input_osw_id]
            else:
                input_list = None
            documentation_object = model.PythonEvaluationProcess(
                label=[
                    model.Label(
                        text="Python Evaluation from Chatbot "
                        + str(datetime.now().strftime("%Y-%m-%d_%H-%M")),
                        lang="en",
                    )
                ],
                input=input_list,
                python_evaluation_code=self.current_python_code,
                uuid=inp.uuid,
                output=[get_full_title(wf)],
                image=get_full_title(wf),
            )
            osw_obj.store_entity(
                OSW.StoreEntityParam(
                    entities=[documentation_object], overwrite=True
                )
            )
            ret_msg += (
                    "documentation object successfully stored: "
                    + str(documentation_object)
            )
            return ret_msg

        except Exception as e:
            return str(e)

    def attach_current_plot_to_osw_page(
        self, inp: AttachCurrentPlotToOswPageInput
    ):
        """uploads the current plot to the osw page with the given id"""
        ret_msg = ""
        try:
            ## get the page object where the plot should be attached

            osw_obj = OswExpress(os.environ.get("OSW_DOMAIN"))
            title = inp.osw_id
            entity = osw_obj.load_entity(title)
            if entity is None:
                return (
                    "error loading entity with title: "
                    + title
                    + " was it formatted correctly?"
                )

            ## save plot to bytesio object:
            if self.plot_panel[0] == self.image_panel:
                bytesio = io.BytesIO(self.image_panel.object)

            elif self.plot_panel[0] == self.matplotlib_panel:
                bytesio = io.BytesIO()
                self.matplotlib_panel.object.savefig(bytesio, format="png")
            else:
                return "no plot to attach"
            plot_uuid = uuid.uuid4()
            wf = WikiFileController(
                uuid=str(plot_uuid),
                osw=osw_obj,
                title="OSW" + str(plot_uuid).replace("-", "") + ".png",
                label=[
                    model.Label(
                        text="Plot from Chatbot "
                        + str(datetime.now().strftime("%Y-%m-%d_%H-%M"))
                    )
                ],
            )
            bytesio.name = wf.title
            try:
                wf.put(bytesio, overwrite=True)
                ret_msg += (
                    "plot successfully uploaded to osw page with uuid: "
                    + str(wf.uuid)
                )
            except Exception as e:
                ret_msg += "error uploading plot to osw page: " + str(e)
            ## link it to the page:
            if not hasattr(entity, "attachments"):
                entity.attachments = []
            entity.attachments.append(get_full_title(wf))
            ## re-upload entity:
            osw_obj.store_entity(
                OSW.StoreEntityParam(entities=[entity], overwrite=True)
            )
            ret_msg += "attachment added successfully, "
            return ret_msg
        except Exception as e:
            return str(e)

    def __panel__(self):
        return self._panel

    def generate_langchain_tools(self) -> List[langchain_core.tools.tool]:
        """
        returns a list of langchain tools that can be called by the agent
        """

        return [  # langchain_core.tools.tool(self.plot_data),
            langchain_core.tools.tool(self.load_data_from_csv),
            langchain_core.tools.tool(self.attach_current_plot_to_osw_page),
            langchain_core.tools.tool(self.plot_by_code),
            langchain_core.tools.tool(self.run_code),
            langchain_core.tools.tool(self.document_current_evaluation),
        ]


### some Agent backend stuff
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class HistoryToolAgent:
    def __init__(self, tools, prompt_template=None):
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
                        "You are a helpful assistant with access to tools. "
                        "ALWAYS call tools directly to fulfill the user's request — "
                        "never describe what a tool call would look like or output JSON "
                        "of a hypothetical call. If you have the information needed to "
                        "call a tool, call it immediately.\n\n"
                        "When the user asks to download files and run code:\n"
                        "1. Download the files using the download tool.\n"
                        "2. Pass the returned file paths directly to plot_by_code or "
                        "run_code via the file_paths parameter.\n"
                        "3. Include all required pip packages in the libraries list.\n\n"
                        "MICRESS / micpy usage (pip package: micress-micpy):\n"
                        "Geometry files (.geof) are automatically paired with binary "
                        "files in the sandbox — do NOT set geometry manually.\n"
                        "Correct API:\n"
                        "  from micpy import bin\n"
                        "  with bin.File('/sandbox/FILENAME.conc1') as f:\n"
                        "      field = f.read(-1)          # NOT read_field()\n"
                        "      fig, ax, cbar = field.plot() # NOT bin.plot()\n"
                        "      fig.savefig('/sandbox/output.png')\n\n"
                        "If the user wants to create something, first try to find the "
                        "category page for the given topic/keyword, then create the instance.",
                    ),
                    ("placeholder", "{chat_history}"),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ]
            )

        # Construct the Tools agent
        self.langchain_agent = create_tool_calling_agent(
            llm, tools, self.prompt_template
        )
        # Create an agent executor by passing in the agent and tools
        self.agent_executor = AgentExecutor(
            agent=self.langchain_agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
        )
        # agent_executor.invoke({"input": "what is LangChain?"})
        self.chat_history = []

    async def invoke(self, prompt):
        res = await self.agent_executor.ainvoke(
            {"input": prompt, "chat_history": self.chat_history}
        )
        self.chat_history.extend(
            [
                HumanMessage(content=prompt),
                str(
                    res["intermediate_steps"]
                ),
                AIMessage(content=res["output"]),
            ]
        )
        return res
