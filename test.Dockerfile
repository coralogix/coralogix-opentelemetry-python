FROM 529726762838.dkr.ecr.eu-west-1.amazonaws.com/alpine/helm:3.12.2 as helm3
FROM 529726762838.dkr.ecr.eu-west-1.amazonaws.com/python:3.11.4-alpine

ENV POETRY_NO_INTERACTION=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
COPY --from=helm3 /usr/bin/helm /usr/local/bin/helm
RUN apk update && apk add build-base git libffi-dev openssl-dev
RUN pip3 install --upgrade pip && pip install poetry==1.5.1
ADD poetry.lock pyproject.toml /app/
WORKDIR /app
RUN poetry config virtualenvs.create false && \
    poetry install
ADD . /app
RUN poetry config virtualenvs.create false && \
    poetry install
ENTRYPOINT ["./scripts/test.sh"]