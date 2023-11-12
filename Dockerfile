# pull python image
FROM python:3.11-slim as base

# set working directory
WORKDIR /app

# set python environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/

# set virtual environment
ENV VIRTUAL_ENV=/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# install system dependencies
RUN apt-get -y update \
  && apt-get -y upgrade \
#   && apt-get -y install python3-distutils \
  && apt-get clean

# use base for build
FROM base as builder

# prevent poetry virtual environment
ENV POETRY_VIRTUALENVS_CREATE false

# install python dependencies
RUN pip install --upgrade pip \
  && pip install --no-cache-dir poetry
COPY ./pyproject.toml ./poetry.lock ./
RUN . /venv/bin/activate && poetry install --no-root --no-interaction --no-ansi --only main

# use base for final image
FROM base as final

# copy built venv folder
COPY --from=builder /venv /venv

# copy src to app folder
COPY . .

# keep container alive for inspection
# CMD sh -c "while true; do sleep 1; done"
