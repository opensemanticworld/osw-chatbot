import sys
import panel as pn


class TerminalMirrorPanel:  ## inspired by https://stackoverflow.com/questions/616645/how-to-duplicate-sys-stdout-to-a-log-file
    """
    a panel that mirrors the terminal output to a panel widget
    """

    def __init__(self):
        self.terminal_panel = pn.widgets.Terminal(
            options={"cursorBlink": True},
            height=300,
            sizing_mode="stretch_width",
        )
        self.stdout = sys.stdout
        sys.stdout = self

    def __panel__(self):
        return self.terminal_panel

    def __del__(self):
        sys.stdout = self.stdout

    def write(self, data):
        self.terminal_panel.write(data)
        self.stdout.write(data)

    def flush(self):
        self.terminal_panel.flush()


if __name__ == "__main__":
    terminal_panel = TerminalMirrorPanel()

    pn.serve(terminal_panel, threaded=True)
