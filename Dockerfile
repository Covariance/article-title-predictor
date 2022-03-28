FROM python:3.9

WORKDIR /workdir

COPY requirements.txt requirements.txt

RUN apt update && apt upgrade -y && pip install --no-cache-dir -r requirements.txt

COPY ./src ./src

COPY ./research/model ./model

ENTRYPOINT [ "streamlit", "run", "src/main.py", "--server.port=8080", "--browser.serverPort=8080" ]
