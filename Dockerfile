FROM python:3.9.17-slim-buster

WORKDIR /app

COPY . .

RUN pip install -r requirement.txt

CMD ["python","churnpredict.py"]

