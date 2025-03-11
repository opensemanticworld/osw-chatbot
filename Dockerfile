#
FROM python:3.11

# 
WORKDIR /app

# 
#RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./requirements.txt /app/requirements.txt
RUN find /app -name "requirements.txt" -type f -exec pip install -r '{}' ';'

# 
COPY ./ /app
RUN pip install .
RUN playwright install-deps && playwright install

# 
#CMD ["uvicorn", "app.src:main_app", "--host", "0.0.0.0", "--port", "80"]
#CMD [ "python3", "app/main.py" ]
#CMD ["uvicorn", "osw_chatbot.structured_output.api:app", "--host", "0.0.0.0", "--port", "80"]
#CMD ["python", "src/osw_chatbot/main.py"]
# options see https://panel.holoviz.org/how_to/server/commandline.html
#CMD ["panel", "serve", "src/osw_chatbot/main.py", "--address", "0.0.0.0", "--port", "81"]