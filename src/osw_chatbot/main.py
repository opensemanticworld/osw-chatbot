
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

from osw_chatbot.toolcalling.agent import invoke

pn.extension()

pn.config.theme = 'dark'


#frontend = ChatFrontendWidget()
from osw_chatbot.toolcalling.agent import frontend, plot_panel
     
async def get_response(contents, user, instance):
    print(contents)
    #frontend.function_call = json.loads(json.dumps({"name": "get_response", "args": [contents]}))
    response = await invoke(contents)
    print(response)
    return response["output"]#["answer"]

def build_app():
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

    app = pn.Row(chat_bot, frontend, plot_panel)
    chat_bot.send("what's on your mind?", user="Assistant", respond=False)
    return app

if __name__ == "__main__":
    pn.serve(build_app, port=52670)