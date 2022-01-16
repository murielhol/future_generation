FROM python:3.7


ENV PYTHONUNBUFFERED=1
RUN pip install pipenv
WORKDIR /script

ADD Pipfile* /script/
ADD src/ /script/src
ADD models /script/models
ADD data /script/data

RUN env
RUN pipenv --python 3.7
RUN pipenv install
ENTRYPOINT ["pipenv", "run", "python", "src/main.py"]
