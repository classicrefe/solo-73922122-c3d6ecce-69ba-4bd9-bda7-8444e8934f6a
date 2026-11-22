FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DB_PATH=/data/nameplate.db
EXPOSE 8000
VOLUME ["/data"]

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "wsgi:app"]
