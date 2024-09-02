FROM python:3.12

WORKDIR /app

RUN curl -sSL https://install.python-poetry.org | python3 -
COPY ./poetry.lock /app/
COPY ./pyproject.toml /app/
RUN /root/.local/bin/poetry export -o ./requests.txt
RUN pip install -r ./requests.txt

COPY ./config.conf /app/
COPY ./analyze.py /app/

ENTRYPOINT ["python", "analyze.py"]
