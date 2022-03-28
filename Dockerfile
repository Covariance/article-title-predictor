FROM python:3.9

WORKDIR /workdir

COPY requirements.txt requirements.txt

RUN apt update && apt upgrade -y && pip install -r requirements.txt

COPY ./src ./src

ENTRYPOINT [ "streamlit", "run", "src/main.py", "--server.port=8080", "--browser.serverPort=8080" ]
