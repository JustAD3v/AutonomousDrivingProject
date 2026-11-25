FROM python:3.11-slim

WORKDIR /app

COPY container-requirements.txt .
RUN pip install --no-cache-dir torch==2.11.0 \ 
    --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r container-requirements.txt

COPY src/ src/

CMD ["python", "src/main.py", "--help"]
