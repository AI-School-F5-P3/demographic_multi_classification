FROM python:3.12-slim

WORKDIR /app

COPY .venv .venv
COPY requirements.txt .
COPY . .

ENV PATH="/app/.venv/bin:$PATH"
RUN pip3 install -r requirements.txt
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]