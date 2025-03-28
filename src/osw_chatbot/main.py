
from pathlib import Path
from dotenv import load_dotenv
import panel as pn

# going to do load_dotenv() here
# as OLLAMA_HOST needs to be in the environment
# before the imports below
env_path = Path(__file__).parent.parent.parent / ".env"
env_loaded = load_dotenv(env_path, verbose=True)
if not env_loaded:
    print(f"No .env file found at {env_path}, using environment variables.")

from osw_chatbot.toolcalling.agent import OswFrontendPanel, PlotToolPanel, HistoryToolAgent
import osw_chatbot.toolcalling.osw_tools as osw_tools
from osw_chatbot.panels.terminal_mirror_panel import TerminalMirrorPanel

pn.extension()

pn.config.theme = 'dark'

def build_app():
    plot_tool_panel = PlotToolPanel()
    #frontend_panel = OswFrontendPanel(child_panels=[plot_tool_panel])
    frontend_panel = OswFrontendPanel(child_panels=[])
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

    async def get_response(contents, user, instance):
        print(contents)
        # frontend.function_call = json.loads(json.dumps({"name": "get_response", "args": [contents]}))
        response = await agent.invoke(contents)
        print(response)
        return response["output"]  # ["answer"]

    chat_bot = pn.chat.ChatInterface(
        callback=get_response,
        #max_height=500,
        #max_width=300,
        show_send=True,
        show_rerun=False,
        show_undo=False,
        show_clear=True,
        show_avatar=True,
        show_timestamp=False,
        show_button_name=False,
        show_reaction_icons=False,
        sizing_mode="stretch_width",
        callback_exception="verbose",
        # stylesheets = [
        #     """
        #     """
        # ],
    )
    terminal_mirror = TerminalMirrorPanel()
    visualization_column = pn.Column( plot_tool_panel, terminal_mirror)
    app = pn.Row(chat_bot, frontend_panel, visualization_column)
    chat_bot.send("what's on your mind?", user="Assistant", respond=False)
    return app

if __name__ == "__main__":
    pn.serve(build_app, port=52670)