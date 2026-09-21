FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY support_assistant/requirements.txt /app/requirements.txt
# Sentence Transformers needs PyTorch. Install its compact CPU-only wheel first
# so pip does not fetch a multi-gigabyte CUDA-enabled distribution.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu \
    && pip install -r requirements.txt
COPY support_assistant /app/support_assistant
EXPOSE 7860
CMD ["uvicorn", "support_assistant.main:app", "--host", "0.0.0.0", "--port", "7860"]
