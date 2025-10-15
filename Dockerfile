#
FROM python:3.11-bookworm

# virtual display for web browser if HEADLESS==false
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y xvfb \
    && apt-get install -qqy x11-apps

# install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 
WORKDIR /app

# 
COPY ./ /app
# init project, install all dependencies
RUN uv sync
# install panel as tool to run it on the cmd line
RUN uv tool install panel
# install panel as tool to run it on the cmd line
RUN uv tool install uvicorn
# needed for "uvicorn", "osw_chatbot.structured_output.api:app"
RUN uv pip install .
# for browser-based web search
RUN uv run playwright install-deps && uv run playwright install 

# wrap commands in xvfb to make display available
ENTRYPOINT ["/bin/sh", "-c", "/usr/bin/xvfb-run -a $@", ""]