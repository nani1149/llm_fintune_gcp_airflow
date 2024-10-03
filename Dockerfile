FROM quay.io/astronomer/astro-runtime:12.1.1
ENV GOOGLE_APPLICATION_CREDENTIALS="C:/gcp/sacred-alliance-433217-e3-54cc0f7cc68f.json"
RUN pip install poetry==1.4.2

WORKDIR /app

COPY sacred-alliance-433217-e3-54cc0f7cc68f.json ./
ENV GOOGLE_APPLICATION_CREDENTIALS=./sacred-alliance-433217-e3-54cc0f7cc68f.json

COPY pyproject.toml poetry.lock ./


RUN poetry install --without dev