FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2-binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy only backend files
COPY cielotuc-backend/cielotuc/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cielotuc-backend/cielotuc/ .

RUN mkdir -p /app/models

EXPOSE 8000

CMD ["gunicorn", "app.main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "2", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120"]
