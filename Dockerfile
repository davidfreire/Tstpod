FROM vllm/vllm-openai:latest

WORKDIR /app

RUN pip install runpod pillow requests decord

COPY handler.py /app/handler.py

CMD ["python","-u","/app/handler.py"]
