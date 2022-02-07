FROM python:3.7


ENV PYTHONUNBUFFERED=1
RUN pip install pipenv
WORKDIR /script

ADD Pipfile* /script/

RUN env
RUN pipenv --python 3.7
RUN pipenv install

ADD src/ /script/src
ADD models /script/models

ENTRYPOINT ["pipenv", "run", "python", "src/main.py"]
