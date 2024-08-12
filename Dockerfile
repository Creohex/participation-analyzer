FROM python:3.12

WORKDIR /app

RUN curl -sSL https://install.python-poetry.org | python3 -

COPY ./analyze.py /app/
COPY ./poetry.lock /app/
COPY ./pyproject.toml /app/

RUN /root/.local/bin/poetry export -o ./requests.txt
RUN pip install -r ./requests.txt

ENTRYPOINT ["python", "analyze.py"]
