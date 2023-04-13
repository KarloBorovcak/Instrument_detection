FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt /app

# RUN pip3 install torch==2.0.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]