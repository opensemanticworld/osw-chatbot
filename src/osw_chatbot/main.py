import panel as pn

from osw_chatbot.toolcalling.agent import (
    HistoryToolAgent,  # PlotToolPanel,
    OswFrontendPanel,
    SandboxPlotPanel,
)

pn.extension()

pn.config.theme = "dark"


def build_app():
    # plot_tool_panel = PlotToolPanel()
    # frontend_panel = OswFrontendPanel(child_panels=[plot_tool_panel])
    sandbox_plot_panel = SandboxPlotPanel()
    frontend_panel = OswFrontendPanel(child_panels=[sandbox_plot_panel])

    tools = [
        # *plot_tool_panel.generate_langchain_tools(),
        *frontend_panel.generate_langchain_tools(),
        *sandbox_plot_panel.generate_langchain_tools(),
    ]

    agent = HistoryToolAgent(tools=tools)

    async def get_response(contents, user, instance):
        print(contents)
        # frontend.function_call = json.loads(json.dumps({"name": "get_response", "args": [contents]})) # noqa
        response = await agent.invoke(contents)
        print(response)
        return response["output"]  # ["answer"]

    chat_bot = pn.chat.ChatInterface(
        callback=get_response,
        # max_height=500,
        # max_width=300,
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

    app = pn.Row(chat_bot, frontend_panel)
    chat_bot.send("what's on your mind?", user="Assistant", respond=False)
    return app


if __name__ == "__main__":
    pn.serve(build_app, port=52670)
