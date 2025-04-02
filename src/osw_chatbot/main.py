from pathlib import Path
from dotenv import load_dotenv
import panel as pn
import param
import warnings

from pydantic import PydanticDeprecatedSince211
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince211)
warnings.filterwarnings("ignore", category=DeprecationWarning, module='pydantic')


# going to do load_dotenv() here
# as OLLAMA_HOST needs to be in the environment
# before the imports below
env_path = Path(__file__).parent.parent.parent / ".env"
env_loaded = load_dotenv(env_path, verbose=True)
if not env_loaded:
    print(f"No .env file found at {env_path}, using environment variables.")

from osw_chatbot.toolcalling.agent import (
    OswFrontendPanel,
    PlotToolPanel,
    HistoryToolAgent,
)
import osw_chatbot.toolcalling.osw_tools as osw_tools
from osw_chatbot.panels.terminal_mirror_panel import TerminalMirrorPanel

pn.extension()

pn.config.theme = "dark"


def build_app():
    plot_tool_panel = PlotToolPanel()
    # frontend_panel = OswFrontendPanel(child_panels=[plot_tool_panel])
    frontend_panel = OswFrontendPanel()
    tools = [
        *plot_tool_panel.generate_langchain_tools(),
        *frontend_panel.generate_langchain_tools(),
        osw_tools.get_page_html,
        osw_tools.download_osl_file,
        osw_tools.get_instances,
        osw_tools.sparql_search_function,
        osw_tools.get_topic_taxonomy,
        osw_tools.find_out_everything_about,
        osw_tools.get_website_html,
        osw_tools.get_file_header,
    ]

    agent = HistoryToolAgent(tools=tools)

    style_chat = """
    .left {
        display: none !important;
    }
    """

    async def get_response(contents, user, instance):
        # print(contents)
        response = await agent.invoke(contents)
        response_message = pn.chat.ChatMessage(
            object=response["output"],
            user="🤖 Assistant",
            show_edit_icon=False,
            show_timestamp=False,
            show_reaction_icons=False,
            stylesheets=[style_chat],
        )
        return response_message  # ["answer"]

    chat_bot = pn.chat.ChatInterface(
        callback=get_response,
        # max_width=500,
        user="🧑 User",
        show_avatar=False,
        min_width=500,
        show_send=True,
        show_rerun=False,
        show_undo=False,
        show_clear=True,
        show_timestamp=False,
        show_button_name=False,
        show_reaction_icons=False,
        callback_exception="verbose",
        # stylesheets=[style_chat],  # not working here, need to be
        # accessed via ChatFeed, where ChatInterface inherits ChatFeed
        # see below to access stylesheets via message_params
        message_params={
            "stylesheets": [style_chat]
        },  # additional params for ChatMessage of user
    )
    terminal_mirror = TerminalMirrorPanel()

    # Responsive Layout with CSS Grid and Media Queries
    media_query = """
    @media screen and (max-width: 768px) {
        div[id^="flexbox"] {
            flex-flow: column !important;
            width: 100vw !important;
            height: auto !important;
        }
    }
    @media screen and (min-width: 769px) {
        div[id^="flexbox"] {
            flex-flow: row nowrap !important;
            height: 100vh !important;
            width: 100vw !important;
        }
    }
    """

    # @media screen and (min-width: 769px) and (max-width: 1200px) {
    #     div[id^="flexbox"] {
    #         flex-flow: row wrap !important;
    #         width: 100vw !important;
    #         height: auto !important;
    #     }
    # }

    chat_bot_pn = pn.Column(
        chat_bot,
        # min_width=500,
        min_height=370,
        max_width=800,
        max_height=900,
    )

    visualization_column = pn.Column(
        pn.Row(
            plot_tool_panel,
        ),
        pn.Row(terminal_mirror),
        # min_height=300,
        min_width=500,
        max_height=900,
        sizing_mode="stretch_width",
    )
    app = pn.FlexBox(
        chat_bot_pn,
        frontend_panel,
        visualization_column,
        stylesheets=[media_query],
    )
    # app = pn.FlexBox(red, green, blue, stylesheets=[media_query]).servable()

    chat_message = pn.chat.ChatMessage(
        object="How can I help you?",
        user="🤖 Assistant",
        show_edit_icon=False,
        show_timestamp=False,
        show_reaction_icons=False,
        stylesheets=[style_chat],
    )

    # chat_bot.send("what's on your mind?", user="Assistant", respond=False)
    chat_bot.send(value=chat_message, respond=False)
    # chat_bot.send(
    #     "Search for a page that has a .csv file attachend, enshure to remember the path of the page, download it, and then plot it. Please show also the URL of the page where the file is located.",
    #     user="User",
    # )

    # Video
    # chat_bot.send(
    #     value="Search for a tensile test",
    #     user="🧑 User",
    #     respond=True,
    # )
    # chat_bot.send(
    #     value="Download the file of page `Demo Tinsile Test`",
    #     user="🧑 User",
    #     respond=True,
    # )
    return app


# pn.extension("terminal")
# build_app().servable()

if __name__ == "__main__":
    pn.serve(build_app, port=52670)
