FROM python:3.9.17-slim-buster

WORKDIR /app

COPY . .

RUN pip install -r requirement.txt

RUN python churnpredict.py

ENTRYPOINT mlflow ui --host="0.0.0.0" --port="5000" 




